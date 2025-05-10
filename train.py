import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import random
import argparse
import logging
from tqdm import tqdm

from models import SimpleEncoder, BackdoorModel
from utils import (similarity_loss, generate_pgd_attack, 
                  visualize_samples, load_trigger_and_target_from_file,
                  calculate_attack_success_rate,
                  generate_classifier_pgd_attack)
from detection import (calculate_embedding_perturbation_score,
                      dynamic_threshold_adaptation, check_trigger_backdoor,
                      multi_objective_optimization)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Trigger Generation ---
def generate_random_trigger(img_shape=(3, 32, 32), pattern_size=(4, 4), intensity=0.3, random_location=True, device='cpu'):
    """
    Generates a random trigger pattern.
    Args:
        img_shape (tuple): Shape of the image (C, H, W).
        pattern_size (tuple): Size of the trigger pattern (h, w).
        intensity (float): Max absolute value of the trigger pixels.
        random_location (bool): If True, place trigger at random location. Otherwise, bottom-right.
        device (str): Device to create tensor on.
    Returns:
        torch.Tensor: The trigger pattern (delta to be added to normalized image).
    """
    C, H, W = img_shape
    patch_h, patch_w = pattern_size
    trigger = torch.zeros(img_shape, device=device)

    # Generate a random pattern for the patch
    random_pattern = (torch.rand(C, patch_h, patch_w, device=device) - 0.5) * 2 * intensity # Values between -intensity and +intensity

    if random_location:
        start_h = random.randint(0, H - patch_h)
        start_w = random.randint(0, W - patch_w)
    else: # Bottom-right corner
        start_h = H - patch_h
        start_w = W - patch_w

    trigger[:, start_h:start_h+patch_h, start_w:start_w+patch_w] = random_pattern
    return trigger

# --- Training Function ---
def train_encoder(model, train_loader, optimizer, criterion, epochs, device,
                  trigger_pattern=None, target_label=None, poison_rate=0.1):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            original_labels = labels.clone() # Keep original labels for accuracy calculation

            if trigger_pattern is not None and target_label is not None:
                # Apply trigger to a subset of the batch
                num_to_poison = int(inputs.size(0) * poison_rate)
                if num_to_poison > 0:
                    poison_indices = torch.randperm(inputs.size(0))[:num_to_poison]
                    
                    # Ensure inputs_triggered is a separate tensor
                    inputs_triggered = inputs.clone()
                    inputs_triggered[poison_indices] += trigger_pattern
                    inputs_triggered[poison_indices] = torch.clamp(inputs_triggered[poison_indices], -1.0, 1.0) # Assuming normalized input to [-1,1] or [0,1]
                                                                                                            # Adjust clamp if normalization is different

                    # Create new labels tensor for poisoned samples
                    labels_with_poison = labels.clone()
                    labels_with_poison[poison_indices] = target_label
                    
                    # Use the modified inputs and labels for these specific samples
                    inputs_for_training = inputs_triggered
                    labels_for_training = labels_with_poison
                else:
                    inputs_for_training = inputs
                    labels_for_training = labels
            else:
                inputs_for_training = inputs
                labels_for_training = labels


            optimizer.zero_grad()
            outputs = model(inputs_for_training)
            loss = criterion(outputs, labels_for_training)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += original_labels.size(0) # Accuracy on original task
            correct_predictions += (predicted == original_labels).sum().item()


        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct_predictions / total_predictions
        if (epoch + 1) % (epochs // min(5, epochs) if epochs > 0 else 1) == 0 or epochs == 1 : # Log a few times per training
            logging.debug(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
    return model


def train_encoder_pgd_adversarial(model, train_loader, optimizer, criterion, epochs, device,
                              epsilon_pgd, steps_pgd, step_size_pgd, 
                              clamp_min=0.0, clamp_max=1.0):
    """
    使用PGD对抗训练方法训练编码器
    """
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (PGD Training)"):
            inputs, labels = inputs.to(device), labels.to(device)

            # 生成PGD对抗样本
            model.eval()  # 生成对抗样本时暂时设为评估模式
            inputs_adv = generate_classifier_pgd_attack(
                model, inputs, labels, criterion,
                epsilon=epsilon_pgd, num_steps=steps_pgd, step_size=step_size_pgd,
                clamp_min=clamp_min, clamp_max=clamp_max, device=device
            )
            model.train()  # 恢复训练模式

            # 对原始样本和对抗样本进行训练
            optimizer.zero_grad()
            
            # 原始样本的损失
            outputs = model(inputs)
            loss_natural = criterion(outputs, labels)
            
            # 对抗样本的损失
            outputs_adv = model(inputs_adv)
            loss_adv = criterion(outputs_adv, labels)
            
            # 总损失 (可以调整权重)
            loss = 0.5 * loss_natural + 0.5 * loss_adv
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            
            # 计算在对抗样本上的准确率
            _, predicted = torch.max(outputs_adv.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct_predictions / total_predictions
        logging.info(f"PGD Training Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Adv Accuracy: {epoch_acc:.4f}")
    
    return model

# --- Main Script ---
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # CIFAR-10 Data
    # Normalization for CIFAR-10: mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
    # Or simpler [-1, 1] normalization: Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # For trigger generation, we use img_shape (C,H,W) which is (3,32,32) for CIFAR10
    IMG_SHAPE = (3, 32, 32)
    NUM_CLASSES = 10

    try:
        train_dataset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform)
    except Exception as e:
        logging.error(f"Failed to download/load CIFAR10. Please check your internet connection or data_dir. Error: {e}")
        return

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # 创建输出目录
    output_dir = os.path.join(args.output_dir, args.train_mode)
    os.makedirs(output_dir, exist_ok=True)

    # 训练指定数量的模型
    for model_idx in tqdm(range(args.num_models), desc=f"Training {args.train_mode} models"):
        # 初始化模型
        model = SimpleEncoder(num_classes=10, embedding_dim=args.embedding_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()

        # 选择训练方式
        if args.train_mode == 'pgd':
            model = train_encoder_pgd_adversarial(
                model, train_loader, optimizer, criterion, args.epochs_per_encoder, device,
                epsilon_pgd=args.epsilon_pgd, 
                steps_pgd=args.steps_pgd,
                step_size_pgd=args.step_size_pgd,
                clamp_min=-1.0, 
                clamp_max=1.0
            )
        else:
            model = train_encoder(
                model, train_loader, optimizer, criterion, args.epochs_per_encoder, device,
                trigger_pattern=None, 
                target_label=None, 
                poison_rate=0.0
            )

        # 保存模型
        model_path = os.path.join(output_dir, f"model_{model_idx:03d}.pth")
        torch.save(model.state_dict(), model_path)
        logging.info(f"Saved model to {model_path}")

    logging.info(f"训练完成，所有模型已保存至 {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Encoders with/without Triggers for CIFAR-10")
    parser.add_argument('--data_dir', type=str, default='./data/cifar10', help='Directory for CIFAR-10 dataset')
    parser.add_argument('--output_dir', type=str, default='./evaluation_models')
    parser.add_argument('--epochs_per_encoder', type=int, default=10, help='Number of epochs to train each encoder')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Dimension of the encoder output embedding')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    parser.add_argument('--train_mode', type=str, required=True, 
                       choices=['standard', 'pgd'], 
                       help="训练模式:standard(标准训练)或 pgd(对抗训练)")
    parser.add_argument('--num_models', type=int, default=5,
                       help="每种模式生成的模型数量")
    parser.add_argument('--epsilon_pgd', type=float, default=0.03,
                       help="PGD攻击的epsilon值")
    parser.add_argument('--steps_pgd', type=int, default=10,
                       help="PGD攻击的迭代次数")
    parser.add_argument('--step_size_pgd', type=float, default=0.01,
                       help="PGD攻击的单步步长")


    args = parser.parse_args()

    # Set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True # Can slow down, but good for reproducibility
        torch.backends.cudnn.benchmark = False

    main(args)