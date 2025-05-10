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
import torch.nn.functional as F
from utils import (similarity_loss, generate_pgd_attack, 
                  visualize_samples, load_trigger_and_target_from_file,
                  calculate_attack_success_rate,
                  generate_classifier_pgd_attack)

def evaluate_robust_accuracy(model, test_loader, criterion, device, 
                             epsilon_pgd, steps_pgd, step_size_pgd,
                             clamp_min=0.0, clamp_max=1.0): # 假设输入已归一化到 [0,1]
    model.eval()
    robust_correct = 0
    total = 0

    for images, labels in tqdm(test_loader, desc="Evaluating Robust Accuracy"):
        images, labels = images.to(device), labels.to(device)

        adv_images = generate_classifier_pgd_attack(model, images, labels, criterion,
                                                    epsilon=epsilon_pgd, num_steps=steps_pgd, 
                                                    step_size=step_size_pgd,
                                                    clamp_min=clamp_min, clamp_max=clamp_max, device=device)

        outputs = model(adv_images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        robust_correct += (predicted == labels).sum().item()

    robust_accuracy = 100 * robust_correct / total
    logging.info(f'Robust Accuracy (PGD Epsilon: {epsilon_pgd}): {robust_accuracy:.2f}%')
    return robust_accuracy

# 建议添加到评估脚本中
import numpy as np

def evaluate_embedding_stability(model, test_loader, criterion, device, 
                                 epsilon_pgd, steps_pgd, step_size_pgd,
                                 clamp_min=0.0, clamp_max=1.0): # 假设输入已归一化到 [0,1]
    model.eval()
    all_similarities = []

    for images, labels in tqdm(test_loader, desc="Evaluating Embedding Stability"): # labels needed for PGD attack generation
        images, labels = images.to(device), labels.to(device)

        adv_images = generate_classifier_pgd_attack(model, images, labels, criterion,
                                                    epsilon=epsilon_pgd, num_steps=steps_pgd, 
                                                    step_size=step_size_pgd,
                                                    clamp_min=clamp_min, clamp_max=clamp_max, device=device)

        with torch.no_grad():
            embeddings_clean = model.get_embeddings(images)
            embeddings_adv = model.get_embeddings(adv_images)

            # Calculate cosine similarity
            # Ensure embeddings are not all zeros if using cosine similarity
            # Handle potential cases where embeddings could be zero vectors (e.g., if ReLU kills all activations)
            sim = F.cosine_similarity(embeddings_clean, embeddings_adv, dim=-1)
            all_similarities.extend(sim.cpu().tolist())

    mean_similarity = np.mean([s for s in all_similarities if not np.isnan(s)]) # Filter out NaNs if any
    logging.info(f'Mean Embedding Cosine Similarity (Clean vs PGD Epsilon {epsilon_pgd}): {mean_similarity:.4f}')
    return mean_similarity