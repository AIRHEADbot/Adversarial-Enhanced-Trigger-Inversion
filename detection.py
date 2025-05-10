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
import json # For saving/loading multiple threshold values
from utils import (similarity_loss, generate_pgd_attack, 
                  visualize_samples, load_trigger_and_target_from_file,
                  calculate_attack_success_rate,
                  generate_classifier_pgd_attack)

def calculate_embedding_perturbation_score(model, dataloader, trigger, device, max_batches=10):
    """
    Calculates the Embedding Perturbation Score (EPS) for a given trigger.
    EPS = 1.0 - mean(cosine_similarity(embeddings_clean, embeddings_triggered))
    """
    model.eval()
    all_similarities = []
    count = 0
    with torch.no_grad():
        for inputs, _ in dataloader:
            if count >= max_batches and max_batches > 0:
                break
            inputs = inputs.to(device)
            
            embeddings_clean = model.get_embeddings(inputs)
            
            # Apply trigger and clamp to valid image range (assuming normalized inputs)
            triggered_inputs = inputs + trigger.to(device) # Ensure trigger is on the same device
            # Clamping depends on the normalization. If inputs are [-1, 1] after normalization:
            # triggered_inputs = torch.clamp(triggered_inputs, -1.0, 1.0)
            # If inputs are [0, 1] after normalization (less common for ImageNet-style models):
            # triggered_inputs = torch.clamp(triggered_inputs, 0.0, 1.0)
            # For robust clamping, determine min/max from original dataset or use known range.
            # Here, assuming normalization leads to values that can be perturbed by trigger and then clamped.
            # A common normalization is to zero mean, unit variance, then trigger is added.
            # For simplicity, let's assume the model's input range allows direct addition and clamping is handled
            # by the user ensuring trigger magnitudes are appropriate or data is pre/post processed.
            # Or, more safely, use the min/max of the *input batch* if dynamic range is unknown.
            # For now, simple clamp to a common normalized range like [-2, 2] if original is around [-1,1] after norm
            # This clamping needs to be consistent with how triggers are expected to be applied.
            # Let's assume inputs are normalized around 0 and trigger perturbs them.
            # A simple way: clamp based on typical normalized image std devs from 0.
            # E.g. if normalized to N(0,1), most values are within [-3,3].
            # A fixed clamp might be too restrictive or too loose.
            # For now, let's use a broad clamp, assuming trigger values are relatively small.
            min_val, max_val = inputs.min(), inputs.max() # Or use fixed values if known e.g. after normalization
            triggered_inputs = torch.clamp(triggered_inputs, min_val, max_val) # Clamp to observed range or theoretical range

            embeddings_triggered = model.get_embeddings(triggered_inputs)
            
            sim = torch.nn.functional.cosine_similarity(embeddings_clean, embeddings_triggered, dim=-1)
            all_similarities.extend(sim.cpu().tolist())
            count += 1
            
    if not all_similarities:
        logging.warning("No similarities calculated for EPS, possibly empty dataloader or max_batches=0.")
        return 0.0 # Or handle as an error

    mean_similarity = np.mean(all_similarities)
    eps_score = 1.0 - mean_similarity
    return eps_score


def dynamic_threshold_adaptation(model, calibration_loader, reference_trigger, target_class, device,
                                 l1_percentile_benign=90,     # L1: Benign L1s should mostly be below this percentile
                                 l1_ref_factor_upper=1.5,     # L1: tau_l1_upper = min(benign_percentile, ref_l1 * factor)
                                 eps_benign_nsigma=2.0,       # EPS: Base for benign EPS
                                 eps_factor_on_difference=0.4, # EPS: Factor to set threshold between benign_mu and ref_eps (increased for more separation)
                                 asr_benign_nsigma=2.0,       # ASR: Base for benign ASR
                                 asr_factor_on_difference=0.4, # ASR: Factor to set threshold between benign_mu and ref_asr (increased for more separation)
                                 eps_asr_max_batches_calib=10, # Batches for EPS/ASR calib
                                 min_asr_threshold=0.2,       # Minimum ASR threshold to consider an attack successful
                                 min_eps_threshold=0.01      # Minimum EPS threshold to consider a perturbation significant
                                 ):
    thresholds = {}
    model.eval()

    # --- L1 Norm Threshold (Upper Bound for Stealthiness) ---
    l1_benign_scores = []
    if reference_trigger is not None:
        img_shape_for_dummy = reference_trigger.shape
    else:
        # Try to get img_shape from loader if reference_trigger is None
        try:
            sample_input, _ = next(iter(calibration_loader))
            img_shape_for_dummy = sample_input.shape[1:]
            logging.info(f"Inferred img_shape_for_dummy from calibration_loader: {img_shape_for_dummy}")
        except StopIteration:
            logging.warning("Calibration loader is empty. Using fallback img_shape (3,32,32) for L1 benign samples.")
            img_shape_for_dummy = (3, 32, 32) # Fallback


    num_l1_benign_samples = 50 # Increased samples for better L1 stats
    for _ in range(num_l1_benign_samples):
        # More diverse benign noise for L1
        noise_type = random.choice(['uniform_small', 'gaussian_tiny', 'zeros'])
        if noise_type == 'uniform_small':
            dummy_noise_l1 = (torch.rand(img_shape_for_dummy, device=device) - 0.5) * 0.05 # Small uniform noise
        elif noise_type == 'gaussian_tiny':
            dummy_noise_l1 = torch.randn(img_shape_for_dummy, device=device) * 0.01 # Tiny Gaussian noise
        else:
            dummy_noise_l1 = torch.zeros(img_shape_for_dummy, device=device)
        l1_benign_scores.append((dummy_noise_l1.abs().sum() / dummy_noise_l1.numel()).item())

    tau_l1_upper_from_benign = np.percentile(l1_benign_scores, l1_percentile_benign) if l1_benign_scores else float('inf')
    logging.info(f"Benign L1s ({num_l1_benign_samples} samples): {l1_percentile_benign}th percentile = {tau_l1_upper_from_benign:.6f}")

    if reference_trigger is not None and reference_trigger.numel() > 0:
        l1_norm_ref = (reference_trigger.abs().sum() / reference_trigger.numel()).item()
        thresholds['l1_norm_reference'] = l1_norm_ref
        logging.info(f"Reference Trigger L1 Norm: {l1_norm_ref:.6f}")
        tau_l1_upper_from_ref = l1_norm_ref * l1_ref_factor_upper
        # The L1 threshold should ideally be determined by what's considered "stealthy" (low L1).
        # A backdoored trigger's L1 should be *below* this.
        # So, tau_l1_upper should be slightly above typical benign noise,
        # and also allow the reference trigger if it's meant to be stealthy.
        # We pick a value that is not excessively large.
        tau_l1_upper = max(tau_l1_upper_from_benign, 0.001) # Ensure it's not zero if benign scores are all zero
        if l1_norm_ref < tau_l1_upper * 1.5 : # If ref trigger L1 is close to or below benign percentile
             tau_l1_upper = min(tau_l1_upper, tau_l1_upper_from_ref) # Allow some flexibility for ref trigger
        logging.info(f"Calculated tau_l1_upper_from_ref: {tau_l1_upper_from_ref:.6f}")
    else:
        l1_norm_ref = float('nan')
        tau_l1_upper = max(tau_l1_upper_from_benign, 0.001) # Ensure it's not zero
        logging.warning("No reference trigger for L1 norm consideration or reference_trigger is empty.")

    thresholds['tau_l1_upper'] = tau_l1_upper
    logging.info(f"L1 Upper Bound Threshold (tau_l1_upper): {tau_l1_upper:.6f} (Candidate L1 should be < this)")

    # --- EPS & ASR Common Reference Calculations ---
    if reference_trigger is None or reference_trigger.numel() == 0: # Also check if numel is zero
        logging.error("Reference trigger is None or empty. Cannot calculate EPS/ASR thresholds properly.")
        thresholds['tau_eps_lower'] = min_eps_threshold # Default to minimum
        thresholds['tau_asr_lower'] = min_asr_threshold # Default to minimum
        return thresholds

    eps_ref_backdoor = calculate_embedding_perturbation_score(model, calibration_loader, reference_trigger, device, max_batches=eps_asr_max_batches_calib)
    asr_ref_backdoor = calculate_attack_success_rate(model, calibration_loader, reference_trigger, target_class, device, max_batches=eps_asr_max_batches_calib)
    thresholds['eps_reference_backdoor'] = eps_ref_backdoor
    thresholds['asr_reference_backdoor'] = asr_ref_backdoor
    logging.info(f"Reference Backdoor EPS: {eps_ref_backdoor:.6f}")
    logging.info(f"Reference Backdoor ASR (target {target_class}): {asr_ref_backdoor:.6f}")

    # Estimate EPS & ASR for benign perturbations
    benign_eps_scores = []
    benign_asr_scores = []
    num_benign_samples_dyn = 50 # Increased samples
    for i in range(num_benign_samples_dyn):
        # 使用不同噪声分布来确保良性样本的多样性
        noise_level_choice = random.random()
        if i % 4 == 0: # Very small uniform noise
            dummy_trigger = (torch.rand(img_shape_for_dummy, device=device) - 0.5) * 0.01
        elif i % 4 == 1: # Small Gaussian noise
            dummy_trigger = torch.randn(img_shape_for_dummy, device=device) * 0.005
        elif i % 4 == 2: # Slightly larger localized noise patch
            dummy_trigger = torch.zeros(img_shape_for_dummy, device=device)
            h, w = img_shape_for_dummy[1:]
            ph, pw = max(1, h // 8), max(1, w // 8)
            sh, sw = random.randint(0, h - ph), random.randint(0, w - pw)
            dummy_trigger[:, sh:sh+ph, sw:sw+pw] = (torch.rand(img_shape_for_dummy[0], ph, pw, device=device) - 0.5) * 0.05
        else: # Zero trigger
            dummy_trigger = torch.zeros(img_shape_for_dummy, device=device)

        eps_dummy = calculate_embedding_perturbation_score(model, calibration_loader, dummy_trigger, device, max_batches=eps_asr_max_batches_calib)
        # For benign ASR, target_class for a random noise should be low for any class.
        # We can average ASR over a few random target classes for a more robust benign ASR estimate,
        # or stick to the main target_class. Sticking to main target_class for now.
        asr_dummy = calculate_attack_success_rate(model, calibration_loader, dummy_trigger, target_class, device, max_batches=eps_asr_max_batches_calib)
        benign_eps_scores.append(eps_dummy)
        benign_asr_scores.append(asr_dummy)

    mu_eps_benign = np.mean(benign_eps_scores) if benign_eps_scores else 0
    std_eps_benign = np.std(benign_eps_scores) if benign_eps_scores else 1e-6
    std_eps_benign = max(std_eps_benign, 1e-6) # Avoid zero std
    logging.info(f"Benign EPS ({num_benign_samples_dyn} samples): mean={mu_eps_benign:.6f}, std={std_eps_benign:.6f}")

    mu_asr_benign = np.mean(benign_asr_scores) if benign_asr_scores else 0
    std_asr_benign = np.std(benign_asr_scores) if benign_asr_scores else 1e-6
    std_asr_benign = max(std_asr_benign, 1e-6) # Avoid zero std
    logging.info(f"Benign ASR ({num_benign_samples_dyn} samples, target {target_class}): mean={mu_asr_benign:.6f}, std={std_asr_benign:.6f}")

    # --- EPS Lower Bound Threshold (tau_eps_lower) ---
    # Candidate EPS should be > this
    # A good backdoor trigger should have significantly higher EPS than benign noise.
    # And its EPS should be comparable to (though not necessarily exceeding) the reference backdoor.
    if eps_ref_backdoor > mu_eps_benign + eps_benign_nsigma * std_eps_benign: # Ensure ref is sufficiently above benign stats
        # Set threshold as a point between benign mean and reference backdoor EPS
        tau_eps_lower = mu_eps_benign + (eps_ref_backdoor - mu_eps_benign) * eps_factor_on_difference
    else:
        logging.warning(f"Reference backdoor EPS ({eps_ref_backdoor:.6f}) is not significantly higher than "
                        f"benign EPS (mean+std*N = {mu_eps_benign + eps_benign_nsigma * std_eps_benign:.6f}). "
                        f"EPS threshold will be based on benign statistics plus a margin, or min_eps_threshold.")
        # Fallback: set it slightly above the benign mean + N*std, or a predefined minimum.
        tau_eps_lower = max(mu_eps_benign + eps_benign_nsigma * std_eps_benign, min_eps_threshold)

    # Ensure the threshold is not excessively high (e.g., not higher than 80% of ref_backdoor_eps unless ref is very low)
    # And not lower than a minimum sensible value.
    tau_eps_lower = min(tau_eps_lower, eps_ref_backdoor * 0.8 if eps_ref_backdoor > min_eps_threshold else eps_ref_backdoor * 1.2)
    tau_eps_lower = max(tau_eps_lower, min_eps_threshold) # Ensure it's at least the minimum
    thresholds['tau_eps_lower'] = tau_eps_lower
    logging.info(f"EPS Lower Bound Threshold (tau_eps_lower): {tau_eps_lower:.6f} (Candidate EPS should be > this)")

    # --- ASR Lower Bound Threshold (tau_asr_lower) ---
    # Candidate ASR should be > this
    if asr_ref_backdoor > mu_asr_benign + asr_benign_nsigma * std_asr_benign: # Ensure ref is sufficiently above benign stats
        tau_asr_lower = mu_asr_benign + (asr_ref_backdoor - mu_asr_benign) * asr_factor_on_difference
    else:
        logging.warning(f"Reference backdoor ASR ({asr_ref_backdoor:.6f}) is not significantly higher than "
                        f"benign ASR (mean+std*N = {mu_asr_benign + asr_benign_nsigma * std_asr_benign:.6f}). "
                        f"ASR threshold will be based on benign statistics plus a margin, or min_asr_threshold.")
        tau_asr_lower = max(mu_asr_benign + asr_benign_nsigma * std_asr_benign, min_asr_threshold)

    # Ensure ASR threshold is not excessively high and not below minimum
    # If reference ASR is very high (e.g. >0.9), we still want a reasonable threshold (e.g. 0.5-0.7 of it)
    # If reference ASR is moderate (e.g. 0.5), a threshold of 0.2-0.3 might be fine.
    tau_asr_lower = min(tau_asr_lower, asr_ref_backdoor * 0.7 if asr_ref_backdoor > min_asr_threshold else asr_ref_backdoor * 1.2) # Cap at 70% of ref ASR
    tau_asr_lower = max(tau_asr_lower, min_asr_threshold) # Ensure at least min_asr_threshold
    tau_asr_lower = min(1.0, tau_asr_lower) # Cap at 1.0

    thresholds['tau_asr_lower'] = tau_asr_lower
    logging.info(f"ASR Lower Bound Threshold (tau_asr_lower): {tau_asr_lower:.6f} (Candidate ASR should be > this for target {target_class})")

    return thresholds

def check_trigger_backdoor(model_to_use, # Can be the global model or a specific loaded one
                           test_loader,
                           candidate_trigger_path, # Path to the .pth file (might contain trigger and target_label)
                           thresholds,
                           default_target_class, # Fallback target_class if not in trigger file
                           device,
                           debug=False,
                           eps_asr_max_batches_check=-1):
    try:
        logging.info(f"---------- Analyzing: {os.path.basename(candidate_trigger_path)} ----------")
        
        # Load trigger and potentially its specific target_label from the candidate file
        candidate_trigger, specific_target_label = load_trigger_and_target_from_file(candidate_trigger_path, device)

        if candidate_trigger is None:
            logging.error(f"Could not extract trigger from {candidate_trigger_path}")
            return False, {"l1_norm": -1, "eps_score": -1, "asr_score": -1, "error": "Load failed", "final_target_class": default_target_class}

        # Determine the target class to use for ASR
        # Priority: 1. specific_target_label from file, 2. default_target_class from args
        final_target_class_for_asr = default_target_class
        if specific_target_label is not None:
            final_target_class_for_asr = specific_target_label
            if specific_target_label != default_target_class:
                logging.info(f"  Using target_label {specific_target_label} from trigger file (overrides default {default_target_class}).")
        else:
            logging.info(f"  No target_label in trigger file, using default_target_class: {default_target_class} for ASR.")

        metrics = {"final_target_class": final_target_class_for_asr}
        
        # 1. L1 Norm
        l1_norm_metric = (candidate_trigger.abs().sum() / candidate_trigger.numel()).item()
        metrics['l1_norm'] = l1_norm_metric
        tau_l1_upper = thresholds.get('tau_l1_upper', float('inf'))
        cond_l1 = l1_norm_metric < tau_l1_upper
        
        # Ensure model is on the correct device and in eval mode
        model_to_use.to(device)
        model_to_use.eval()

        # 2. EPS
        eps_score = calculate_embedding_perturbation_score(model_to_use, test_loader, candidate_trigger, device, max_batches=eps_asr_max_batches_check)
        metrics['eps_score'] = eps_score
        tau_eps_lower = thresholds.get('tau_eps_lower', 0.0)
        cond_eps = eps_score > tau_eps_lower

        # 3. ASR
        # Use final_target_class_for_asr for ASR calculation
        asr_score = calculate_attack_success_rate(model_to_use, test_loader, candidate_trigger, final_target_class_for_asr, device, max_batches=eps_asr_max_batches_check)
        metrics['asr_score'] = asr_score
        tau_asr_lower = thresholds.get('tau_asr_lower', 0.0)
        cond_asr = asr_score > tau_asr_lower

        is_backdoor = cond_l1 and cond_eps and cond_asr
        
        logging.info(f"  L1 Norm  : {metrics['l1_norm']:.6f} (Condition: < {tau_l1_upper:.6f}) -> {cond_l1}")
        logging.info(f"  EPS Score: {metrics['eps_score']:.6f} (Condition: > {tau_eps_lower:.6f}) -> {cond_eps}")
        logging.info(f"  ASR Score: {metrics['asr_score']:.6f} (Condition: > {tau_asr_lower:.6f}) -> {cond_asr} (target: {final_target_class_for_asr})")
        logging.info(f"  Overall Backdoor Detected: {is_backdoor}")
            
        return is_backdoor, metrics
        
    except Exception as e:
        logging.error(f"Error processing trigger {candidate_trigger_path}: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return False, {"l1_norm": -1, "eps_score": -1, "asr_score": -1, "error": str(e), "final_target_class": default_target_class}



def multi_objective_optimization(model, inputs, img_shape, target_class=0, lambda1=0.8, lambda2=0.5, lambda3=1.0, 
                                 epochs=100, lr=0.01, epsilon=8/255, device='cpu', visualize_every=20):
    """
    Optimize a trigger pattern for backdoor detection
    
    Args:
        model: The model to optimize against
        inputs: Batch of input samples
        img_shape: Shape of input images (C,H,W)
        target_class: Target class for the attack
        lambda1-3: Loss component weights
        epochs: Number of optimization epochs
        lr: Learning rate
        epsilon: Maximum perturbation magnitude for adversarial examples
        device: Computing device
        visualize_every: How often to visualize progress
        
    Returns:
        Optimized trigger pattern
    """
    # Initialize trigger with small random values for better optimization
    trigger = nn.Parameter(torch.randn(img_shape, device=device) * 0.05)
    optimizer = torch.optim.Adam([trigger], lr=lr)
    
    # Cosine annealing scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    criterion = nn.CrossEntropyLoss()
    
    # Use tqdm for progress tracking
    progress_bar = tqdm(range(epochs), desc="Optimizing trigger")
    
    for epoch in progress_bar:
        model.train()  # Ensure model is in training mode for optimization
        optimizer.zero_grad()

        # Get clean embeddings
        embeddings_clean = model.get_embeddings(inputs)
        
        # Apply trigger and get triggered embeddings
        triggered_inputs = inputs + trigger
        triggered_inputs = torch.clamp(triggered_inputs, 0, 1)  # Ensure valid image values
        embeddings_triggered = model.get_embeddings(triggered_inputs)
        
        # 1. Stealth loss - L1 norm of trigger normalized by number of elements
        stealth_loss = trigger.abs().sum() / trigger.numel()
        
        # 2. Clean similarity loss - penalize dissimilarity between clean and triggered embeddings
        clean_similarity = similarity_loss(embeddings_clean, embeddings_triggered)
        clean_similarity_loss = 1 - clean_similarity
        
        # 3. Attack effectiveness loss - make the model predict target class
        outputs_triggered = model(triggered_inputs)
        attack_effectiveness_loss = criterion(outputs_triggered, 
                                          torch.full((inputs.size(0),), target_class, dtype=torch.long, device=device))
        
        # 4. Generate adversarial examples for robustness testing
        adv_inputs = generate_pgd_attack(model, inputs, trigger, epsilon=epsilon)
        
        # Apply trigger to adversarial inputs
        adv_triggered_inputs = adv_inputs + trigger
        adv_triggered_inputs = torch.clamp(adv_triggered_inputs, 0, 1)
        
        # Get embeddings for adversarial triggered inputs
        embeddings_adv_triggered = model.get_embeddings(adv_triggered_inputs)
        
        # Calculate adversarial similarity loss
        adv_similarity = similarity_loss(embeddings_clean, embeddings_adv_triggered)
        adv_similarity_loss = 1 - adv_similarity
        
        # Combined loss with weighted components
        total_loss = (stealth_loss + 
                     lambda1 * clean_similarity_loss + 
                     lambda3 * attack_effectiveness_loss + 
                     lambda2 * adv_similarity_loss)
        
        # Backpropagate and update
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Ensure trigger values stay in reasonable range
        trigger.data.clamp_(-0.5, 0.5)  # Limit trigger magnitude
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{total_loss.item():.4f}',
            'stealth': f'{stealth_loss.item():.4f}',
            'attack': f'{attack_effectiveness_loss.item():.4f}'
        })
        
        # Visualize periodically
        if epoch % visualize_every == 0 and inputs.size(0) > 0:
            os.makedirs('adv_visualization', exist_ok=True)
            
            # Select sample for visualization
            sample_idx = 0
            
            # Get visualization samples
            clean_sample = inputs[sample_idx].detach()
            trigger_pattern = trigger.detach()
            triggered_sample = torch.clamp(clean_sample + trigger_pattern, 0, 1)
            adv_sample = adv_inputs[sample_idx].detach()
            
            visualize_samples(
                clean_sample,
                adv_sample,
                triggered_sample,
                trigger_pattern,
                save_path=f'adv_visualization/epoch_{epoch}.png'
            )
            
            # Log detailed metrics
            logging.info(f'Epoch {epoch}, Total: {total_loss.item():.4f}, '
                         f'Stealth: {stealth_loss.item():.4f}, '
                         f'Clean_Sim: {clean_similarity.item():.4f}, '
                         f'Attack: {attack_effectiveness_loss.item():.4f}, '
                         f'Adv_Sim: {adv_similarity.item():.4f}')
    
    return trigger.detach()