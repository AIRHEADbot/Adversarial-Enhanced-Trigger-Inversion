import torch
import logging
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import os

def similarity_loss(embeddings_clean, embeddings_triggered):
    """Calculate cosine similarity between embeddings"""
    sim = torch.nn.functional.cosine_similarity(embeddings_clean, embeddings_triggered, dim=-1)
    return sim.mean()


def generate_pgd_attack(model, x, trigger, epsilon=8/255, steps=10, step_size=1.5/255):
    """
    Generate adversarial examples using PGD attack
    
    Args:
        model: The model to attack (should have get_embeddings method)
        x: Input samples
        trigger: Trigger pattern to apply
        epsilon: Maximum perturbation magnitude
        steps: Number of PGD steps
        step_size: Step size for PGD
        
    Returns:
        Adversarial examples
    """
    # Store original model state
    original_mode = model.training
    model.eval()  # Ensure model is in eval mode for attack generation
    
    delta = torch.zeros_like(x, requires_grad=True)
    
    for _ in range(steps):
        # Apply current perturbation and ensure values are valid
        adv_input = torch.clamp(x + delta, 0, 1)
        
        # Get embeddings for comparison
        with torch.enable_grad():  # Ensure gradients are enabled for this block
            embeddings_clean = model.get_embeddings(x)
            embeddings_adv_triggered = model.get_embeddings(adv_input + trigger)
            
            # Calculate loss as negative similarity (want to maximize dissimilarity)
            loss = -similarity_loss(embeddings_clean, embeddings_adv_triggered)
            loss.backward()
        
        # Update perturbation with gradient sign method
        delta.data = delta.data - step_size * delta.grad.detach().sign()  # Gradient descent since we negated the loss
        
        # Project perturbation back to epsilon-ball
        delta.data = torch.clamp(delta.data, -epsilon, epsilon)
        
        # Reset gradients
        delta.grad.zero_()
    
    # Restore model's original state
    model.train(original_mode)
    
    # Return perturbed inputs, ensuring values are valid
    return torch.clamp(x + delta.detach(), 0, 1)

def visualize_samples(clean, adv_input, triggered_input, trigger, save_path=None):
    """
    Visualize samples for debugging and progress tracking
    
    Args:
        clean: Clean input sample
        adv_input: Adversarial input sample
        triggered_input: Input with trigger applied
        trigger: Trigger pattern
        save_path: Path to save visualization
    """
    # Create figure
    plt.figure(figsize=(20, 5))
    
    # Helper function for normalization and plotting
    def normalize_and_plot(ax, img, title):
        # Convert to numpy and ensure proper range
        if torch.is_tensor(img):
            img = img.detach().cpu()
            
        # Handle different normalizations
        if img.min() < 0 or img.max() > 1:
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            
        # Convert tensor to numpy for plotting
        if torch.is_tensor(img):
            img = img.permute(1, 2, 0).numpy()
            
        # Clip to ensure proper display
        img = np.clip(img, 0, 1)
        
        # Plot
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    
    # Plot clean sample
    ax1 = plt.subplot(1, 4, 1)
    normalize_and_plot(ax1, clean, 'Clean Sample')
    
    # Plot adversarial sample
    ax2 = plt.subplot(1, 4, 2)
    normalize_and_plot(ax2, adv_input, 'Adversarial Sample')
    
    # Plot triggered sample
    ax3 = plt.subplot(1, 4, 3)
    normalize_and_plot(ax3, triggered_input, 'Triggered Sample')
    
    # Plot trigger pattern
    ax4 = plt.subplot(1, 4, 4)
    # For trigger, normalize specially to highlight pattern
    trigger_vis = trigger.clone().detach().cpu()
    trigger_vis = (trigger_vis - trigger_vis.min()) / (trigger_vis.max() - trigger_vis.min() + 1e-8)
    normalize_and_plot(ax4, trigger_vis, 'Trigger Pattern')
    
    # Save or show
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def load_trigger_and_target_from_file(file_path, device):
    """
    Load trigger and target_label from file.
    Returns:
        (torch.Tensor, int): Trigger tensor, target_label. Or (None, None)
    """
    try:
        loaded_obj = torch.load(file_path, map_location=device)
        trigger_tensor = None
        target_label = None

        if isinstance(loaded_obj, dict):
            if 'trigger' in loaded_obj and isinstance(loaded_obj['trigger'], torch.Tensor):
                trigger_tensor = loaded_obj['trigger']
            if 'target_label' in loaded_obj: # Common key for target label
                target_label = int(loaded_obj['target_label'])

            # Fallback if 'trigger' key not found but other common keys exist
            if trigger_tensor is None:
                common_keys = ['t', 'backdoor', 'perturbation', 'delta', 'pattern', 'mask']
                for key in common_keys:
                    if key in loaded_obj and isinstance(loaded_obj[key], torch.Tensor):
                        logging.info(f"Found trigger in key '{key}'")
                        trigger_tensor = loaded_obj[key]
                        break
            # If still no specific target_label, but there's a 'label' or 'class'
            if target_label is None:
                for key in ['label', 'class', 'target']:
                    if key in loaded_obj:
                        try:
                            target_label = int(loaded_obj[key])
                            logging.info(f"Found target label in key '{key}'")
                            break
                        except ValueError:
                            logging.warning(f"Could not convert value for key '{key}' to int.")


        elif isinstance(loaded_obj, torch.Tensor): # If the file just contains the trigger tensor
            trigger_tensor = loaded_obj
            # Target label would be unknown from this format alone, would need to be passed externally

        elif isinstance(loaded_obj, nn.Parameter):
            trigger_tensor = loaded_obj.data
            # Target label also unknown

        if trigger_tensor is None:
            logging.warning(f"Could not extract a tensor suitable for a trigger from {file_path}")
            return None, None
        
        # If target_label is still None here, it means it wasn't found in the dict or the file wasn't a dict.
        # The caller will have to use a default or args.target_class.
        # For backdoor_trigger_XXX.pth files, target_label should be present.

        return trigger_tensor, target_label

    except Exception as e:
        logging.error(f"Failed to load trigger/target from {file_path}: {str(e)}")
        return None, None

def calculate_attack_success_rate(model, dataloader, trigger, target_class, device, max_batches=10):
    """
    Calculates the Attack Success Rate (ASR) for a given trigger and target class.
    """
    model.eval()
    correct_targeted_predictions = 0
    total_samples = 0
    count = 0
    with torch.no_grad():
        for inputs, _ in dataloader: # Labels are not used for ASR against a fixed target
            if count >= max_batches and max_batches > 0:
                break
            inputs = inputs.to(device)
            
            triggered_inputs = inputs + trigger.to(device)
            # Apply appropriate clamping based on your input normalization
            min_val, max_val = inputs.min(), inputs.max() # Example clamping
            triggered_inputs = torch.clamp(triggered_inputs, min_val, max_val)

            outputs = model(triggered_inputs)
            _, predicted_classes = torch.max(outputs, 1)
            
            correct_targeted_predictions += (predicted_classes == target_class).sum().item()
            total_samples += inputs.size(0)
            count += 1
            
    if total_samples == 0:
        logging.warning("No samples processed for ASR calculation.")
        return 0.0
    
    asr = correct_targeted_predictions / total_samples
    return asr



def generate_classifier_pgd_attack(model, x_natural, y_true, criterion, 
                                  epsilon=8/255, num_steps=10, step_size=2/255, 
                                  clamp_min=0.0, clamp_max=1.0, device='cpu'):
    """
    使用投影梯度下降（PGD）生成对抗样本
    
    Args:
        model: 要攻击的模型
        x_natural: 干净的输入样本
        y_true: 真实标签
        criterion: 损失函数
        epsilon: 最大扰动范围
        num_steps: PGD步骤数量
        step_size: 每步PGD的步长
        clamp_min: 输入的最小值
        clamp_max: 输入的最大值
        device: 计算设备
        
    Returns:
        对抗样本
    """
    # 确保模型处于评估模式
    model.eval()
    
    # 将输入数据移动到指定设备
    x_natural = x_natural.detach().to(device)
    y_true = y_true.detach().to(device)
    
    # 初始化对抗样本为原始样本的副本
    x_adv = x_natural.clone()
    
    # 在epsilon球内随机初始化扰动
    # 注意：我们不直接使用delta张量，而是直接操作x_adv
    if epsilon > 0:
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
        # 确保初始扰动后的样本仍在有效范围内
        x_adv = torch.clamp(x_adv, clamp_min, clamp_max)
    
    for _ in range(num_steps):
        # 每次迭代都重新创建计算图
        x_adv.requires_grad_(True)
        
        # 前向传播
        outputs = model(x_adv)
        model.zero_grad()  # 确保模型梯度清零
        
        # 计算损失（注意这里我们要最大化损失，所以在后面取负梯度）
        loss = criterion(outputs, y_true)
        
        # 计算输入的梯度
        loss.backward()
        
        # 获取梯度
        grad = x_adv.grad.detach()
        
        # 确保x_adv不再需要梯度，以便我们可以更新它
        x_adv = x_adv.detach()
        
        # 更新x_adv（朝着梯度的方向移动，使损失增加）
        x_adv = x_adv + step_size * grad.sign()
        
        # 将扰动限制在epsilon球内
        # 首先计算总扰动
        delta = x_adv - x_natural
        delta = torch.clamp(delta, -epsilon, epsilon)
        
        # 使用限制后的扰动更新x_adv
        x_adv = x_natural + delta
        
        # 确保x_adv在有效范围内
        x_adv = torch.clamp(x_adv, clamp_min, clamp_max)
    
    # 确保返回的张量不附带梯度信息
    return x_adv.detach()