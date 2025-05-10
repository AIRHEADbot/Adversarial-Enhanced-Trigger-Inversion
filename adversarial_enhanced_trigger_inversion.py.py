import torch
import torch.nn as nn
import numpy as np
import argparse
import logging
import os
import matplotlib.pyplot as plt  
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.utils import save_image 

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 定义相似性损失函数
def similarity_loss(embeddings_clean, embeddings_triggered):
    """
    计算嵌入的余弦相似度均值
    :param embeddings: 嵌入向量
    :return: 余弦相似度均值
    """
    sim = torch.nn.functional.cosine_similarity(embeddings_clean, embeddings_triggered, dim=-1)
    return sim.mean()

# 对抗样本生成函数
def generate_pgd_attack(model, x, trigger, epsilon=8/255, steps=10, step_size=1.5/255):
    """
    使用PGD方法生成对抗样本
    :param model: 模型
    :param x: 输入数据
    :param trigger: 触发器
    :param epsilon: 扰动幅度
    :param steps: 迭代步数
    :param step_size: 步长
    :return: 对抗样本
    """
    delta = torch.zeros_like(x, requires_grad=True)
    for _ in range(steps):
        adv_input = torch.clamp(x + delta, 0, 1)
        embeddings_clean = model(x)
        embeddings_adv = model(adv_input + trigger)
        loss = similarity_loss(embeddings_clean, embeddings_adv)
        loss.backward()
        delta.data = delta.data + step_size * delta.grad.detach().sign()
        delta.data = torch.clamp(delta.data, -epsilon, epsilon)
        delta.grad.zero_()
    return torch.clamp(x + delta.detach(), 0, 1)

# 多目标优化函数
def multi_objective_optimization(model, inputs,  img_shape, target_class=0, lambda1=0.8, lambda2=0.5, lambda3=1.0, epochs=100, lr=0.01, epsilon=8/255):
    """
    多目标优化触发器
    :param model: 模型
    :param inputs: 输入数据
    :param img_shape: 图像形状
    :param lambda1: 干净样本相似性损失权重
    :param lambda2: 对抗样本相似性损失权重
    :param epochs: 训练轮数
    :param lr: 学习率
    :return: 优化后的触发器
    """
    # 初始化trigger
    trigger = nn.Parameter(torch.zeros(img_shape).to(inputs.device))
    trigger.data[:, :4, :4] = 0.1  
    # trigger.data[:, :8, :8] = 0.05
    optimizer = torch.optim.Adam([trigger], lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        #triggered_inputs = (1 - m) * inputs + m * trigger
        embeddings_clean = model(inputs)
        #embeddings_triggered = model(triggered_inputs)
        embeddings_triggered = model(inputs + trigger)
        clean_loss = trigger.norm(p=1) + lambda1 * similarity_loss(embeddings_clean, embeddings_triggered)
        
        outputs_triggered = model(inputs + trigger)
        attack_loss = lambda3 * criterion(outputs_triggered, torch.full((inputs.size(0),), target_class, device=inputs.device))
        
        adv_inputs = generate_pgd_attack(model, inputs, trigger, epsilon=epsilon)
        # 获取当前batch的文件名
        # _, _, filenames = next(iter(train_loader)) # 这行代码会导致错误，因为train_loader没有在函数内部定义。
        # save_adversarial_batch(adv_inputs, filenames) #这行代码也会导致错误，因为save_adversarial_batch没有定义
        embeddings_adv = model(adv_inputs)
        adv_loss = lambda2 * similarity_loss(embeddings_clean, embeddings_adv)

        # 梯度归一化并更新
        total_loss = clean_loss + attack_loss + adv_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            sample_idx = 0  # 选择第一个样本
            clean_sample = inputs[sample_idx].detach()
            adv_sample = adv_inputs[sample_idx].detach()
            current_trigger = trigger.detach()
        
            # 确保目录存在
            os.makedirs('adv_visualization', exist_ok=True)
            visualize_samples(
                clean_sample,
                adv_sample,
                current_trigger,
                save_path=f'adv_visualization/epoch_{epoch}.png'
            )
    
        if epoch % 10 == 0:
            logging.info(f'Epoch {epoch}, Total Loss: {total_loss.item()}, Clean: {clean_loss.item()}, Attack: {attack_loss.item()}, Adv: {adv_loss.item()}')
    
    return trigger

# 动态阈值调整函数
def dynamic_threshold_adaptation(model, calibration_dataset, trigger, device):
    """
    动态调整检测阈值
    :param model: 模型
    :param calibration_dataset: 校准数据集
    :param trigger: 触发器
    :param device: 设备
    :return: 动态阈值
    """
    """
    pl1_norms = []
    calibration_loader = DataLoader(calibration_dataset, batch_size=1, shuffle=False)
    for batch in calibration_loader:
        inputs = batch[0].to(device)
        pl1 = trigger.abs().sum() / trigger.numel()
        pl1_norms.append(pl1.item())
    """
    pl1 = trigger.abs().sum() / trigger.numel()
    pl1_norms = [pl1.item()] * len(calibration_dataset)
    mu_adv = np.mean(pl1_norms)
    sigma_adv = np.std(pl1_norms)
    tau = mu_adv + 3 * sigma_adv
    #tau = np.percentile(pl1_norms, 99.7)
    return tau

def visualize_samples(clean, adv, trigger, save_path=None):
    """
    可视化清洁样本、对抗样本与触发器
    :param clean: 原始样本张量 (C,H,W)
    :param adv: 对抗样本张量
    :param trigger: 触发器张量
    :param save_path: 图像保存路径
    """
    # 反归一化处理
    clean = denorm(clean)
    adv = denorm(adv)
    trigger = (trigger - trigger.min()) / (trigger.max() - trigger.min())
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(clean.permute(1, 2, 0).cpu().numpy())
    plt.title('Clean Sample')
    
    plt.subplot(1, 3, 2)
    plt.imshow(adv.permute(1, 2, 0).cpu().numpy())
    plt.title('Adversarial Sample')
    
    plt.subplot(1, 3, 3)
    plt.imshow(trigger.permute(1, 2, 0).cpu().numpy())
    plt.title('Trigger Pattern')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def save_adversarial_batch(adv_batch, indices, output_dir='adv_samples'):
    """
    批量保存对抗样本为PNG文件
    :param adv_batch: 对抗样本张量 (B,C,H,W)
    :param filenames: 原始文件名列表
    :param output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    adv_batch = denorm(adv_batch)  # 反归一化到[0,1]
    
    for i in range(adv_batch.size(0)):
        img = transforms.ToPILImage()(adv_batch[i].cpu())
        
        save_path = os.path.join(output_dir, f'adv_image_{indices[i]}.png')
        img.save(save_path)

def denorm(x):
    """Denormalize tensor from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp(0, 1)

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Adversarially Enhanced Trigger Inversion')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--lambda1', type=float, default=1.0)
    parser.add_argument('--lambda2', type=float, default=0.5)
    parser.add_argument('--lambda3', type=float, default=0.9)
    parser.add_argument('--epsilon', type=float, default=12/255)
    args = parser.parse_args()

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # train_dataset = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform)
    class CIFAR10WithNames(datasets.CIFAR10):
        def __getitem__(self, index):
            img, target = super().__getitem__(index)
            return img, target, index
        
    train_dataset = CIFAR10WithNames(root=args.data_dir, train=True, download=True, transform=transform) # 替换为带文件名的dataset
    calibration_dataset = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    class SSLEncoder(nn.Module):
        def __init__(self):
            super(SSLEncoder, self).__init__()
            self.resnet = models.resnet18(weights=None)
            # self.resnet.fc = nn.Linear(512, 128)
            self.resnet.fc = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128)
            )
        def forward(self, x):
            return self.resnet(x)
    
    model = SSLEncoder().to(device)

    # 加载输入数据
    inputs, _, filenames = next(iter(train_loader)) 
    inputs = inputs.to(device)
    img_shape = inputs.shape[1:]
    m = torch.zeros(img_shape).to(device)
    m[:, -4:, -4:] = 1.0 # 掩码区域为右下角 4x4 像素

    # 多目标优化
    optimized_trigger = multi_objective_optimization(model, inputs, img_shape,
                                                     lambda1=args.lambda1, lambda2=args.lambda2, lambda3=args.lambda3,
                                                     epochs=args.epochs, lr=args.lr, epsilon=args.epsilon)
    
    # 动态阈值调整
    tau = dynamic_threshold_adaptation(model, calibration_dataset, optimized_trigger, device)

    # 保存优化后的触发器和动态阈值
    torch.save(optimized_trigger, 'optimized_trigger.pth')
    '''torch.save(
    {'m': m, 't': optimized_trigger},  
    'optimized_trigger_2.pth'
    )'''
    with open('dynamic_threshold.txt', 'w') as f:
        f.write(str(tau))
    
    #logging.info(f"Optimized trigger shape: {optimized_trigger.shape}")
    #logging.info(f"Dynamic threshold: {tau}")
    
    # 在main函数中保存对抗样本
    inputs, _, indices = next(iter(train_loader))  # 获取数据和索引
    inputs = inputs.to(device)
    adv_inputs = generate_pgd_attack(model, inputs, optimized_trigger, epsilon=args.epsilon)
    # 使用批次中的样本索引作为文件名标识
    save_adversarial_batch(adv_inputs, list(range(len(indices))), output_dir='adv_samples')# 保存对抗样本
    
if __name__ == "__main__":
    main()
