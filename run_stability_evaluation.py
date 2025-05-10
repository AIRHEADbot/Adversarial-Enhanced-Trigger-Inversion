import torch
import torch.nn as nn
import numpy as np
import argparse
import logging
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import glob # For finding trigger files
import random
from tqdm import tqdm  # For progress bar
import json

# 导入拆分后的模块
from models import SimpleEncoder, BackdoorModel
from utils import (similarity_loss, generate_pgd_attack, 
                  visualize_samples, load_trigger_and_target_from_file,
                  calculate_attack_success_rate,
                  generate_classifier_pgd_attack)
from detection import (calculate_embedding_perturbation_score,
                      dynamic_threshold_adaptation, check_trigger_backdoor,
                      multi_objective_optimization)
from evaluation import (evaluate_robust_accuracy, evaluate_embedding_stability)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main_eval():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- 数据加载参数 ---
    data_dir = './data/cifar10' # 根据你的数据集调整
    batch_size = 64
    # CIFAR-10 归一化，确保与训练时一致
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 与 generate_test_triggers.py 中一致
    ])
    # 加载测试集
    try:
        test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    except Exception as e:
        logging.error(f"Failed to load CIFAR10 test set. Error: {e}")
        return
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # --- PGD 攻击参数 (用于评估，应与训练时的对抗扰动强度相似或略有不同以测试泛化性) ---
    epsilon_pgd_eval = 8/255
    steps_pgd_eval = 20 # 评估时可以多几步
    step_size_pgd_eval = 2/255
    clamp_min_eval = -1.0 # CIFAR-10 Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) 后范围是 [-1, 1]
    clamp_max_eval = 1.0

    # --- 模型加载 ---
    # 假设 SimpleEncoder(num_classes=10, embedding_dim=128)
    num_classes = 10 # CIFAR-10
    embedding_dim = 128 # 与 generate_test_triggers.py 中 SimpleEncoder 的默认值一致
    criterion = nn.CrossEntropyLoss()

    base_dir = "evaluation_models"

    model_groups = {
        "PGD": os.path.join(base_dir, "pgd", "*.pth"),
        "Standard": os.path.join(base_dir, "standard", "*.pth")
    }
    model_list = []
    for group_name, pattern in model_groups.items():
        model_paths = glob.glob(pattern)
        for path in model_paths:
            model_list.append({
                "group": group_name,
                "path": path,
                "name": os.path.basename(path).split('.')[0]
            })

    if not model_list:
        logging.error("No models found in specified directories")
        return

    

    results = []

    for model_info in tqdm(model_list, desc="Evaluating models"):
        model = SimpleEncoder(num_classes=num_classes, embedding_dim=embedding_dim).to(device)
        
        try:
            model.load_state_dict(torch.load(model_info["path"], map_location=device))
        except Exception as e:
            logging.error(f"Failed to load {model_info['path']}: {e}")
            continue
        model.eval()

        # 评估鲁棒准确率
        robust_acc = evaluate_robust_accuracy(
            model, test_loader, criterion, device,
            epsilon_pgd_eval, steps_pgd_eval, step_size_pgd_eval,
            clamp_min_eval, clamp_max_eval
        )
        # 评估嵌入稳定性
        embedding_sim = evaluate_embedding_stability(
            model, test_loader, criterion, device,
            epsilon_pgd_eval, steps_pgd_eval, step_size_pgd_eval,
            clamp_min_eval, clamp_max_eval
        )

        results.append({
            "group": model_info["group"],
            "name": model_info["name"],
            "robust_accuracy": robust_acc,
            "embedding_similarity": embedding_sim
        })
        logging.info(
            f"{model_info['group']} - {model_info['name']}: "
            f"Robust Acc = {robust_acc:.2f}%, "
            f"Embedding Sim = {embedding_sim:.4f}"
        )
    # --- 论证 PGD 提升稳定性 ---
    summary = {
        "PGD": {"robust_acc": [], "embed_sim": []},
        "Standard": {"robust_acc": [], "embed_sim": []}
    }

    for res in results:
        summary[res["group"]]["robust_acc"].append(res["robust_accuracy"])
        summary[res["group"]]["embed_sim"].append(res["embedding_similarity"])

    # --- 可视化结果 ---
    plt.figure(figsize=(12, 6))

    # 鲁棒准确率比较
    plt.subplot(1, 2, 1)
    for group in ["PGD", "Standard"]:
        if summary[group]["robust_acc"]:
            avg = np.mean(summary[group]["robust_acc"])
            plt.bar(group, avg, alpha=0.7, label=f"{group} (Avg: {avg:.2f}%)")
    plt.title("Robust Accuracy Comparison")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    # 嵌入相似度比较
    plt.subplot(1, 2, 2)
    for group in ["PGD", "Standard"]:
        if summary[group]["embed_sim"]:
            avg = np.mean(summary[group]["embed_sim"])
            plt.bar(group, avg, alpha=0.7, label=f"{group} (Avg: {avg:.4f})")
    plt.title("Embedding Similarity Comparison")
    plt.ylabel("Similarity")
    plt.legend()

    plt.tight_layout()
    plot_filename = "batch_evaluation_comparison.png"
    plt.savefig(plot_filename)
    logging.info(f"Comparison plot saved to {plot_filename}")

    # 保存详细结果
    result_filename = "batch_evaluation_results.json"
    with open(result_filename, "w") as f:
        json.dump({
            "summary": summary,
            "detailed": results
        }, f, indent=2)
    
    logging.info(f"Detailed results saved to {result_filename}")

    return results
    


if __name__ == '__main__':
    evaluation_results = main_eval()
    print("\nFinal Evaluation Summary:")
    for res in evaluation_results:
        print(f"[{res['group']}] {res['name']}:")
        print(f"  Robust Accuracy: {res['robust_accuracy']:.2f}%")
        print(f"  Embedding Similarity: {res['embedding_similarity']:.4f}\n")