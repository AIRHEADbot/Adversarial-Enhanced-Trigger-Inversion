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

def main():
    """Main function implementing the backdoor detection tool"""
    parser = argparse.ArgumentParser(description='Enhanced Backdoor Trigger Detection Tool')
    
    # Operation mode
    parser.add_argument('--mode', type=str, required=True, 
                      choices=['generate_reference', 'detect_backdoors', 'analyze_trigger'], 
                      help='Operation mode')
    
    # Data and model parameters
    parser.add_argument('--data_dir', type=str, default='./data', 
                      help='Directory for datasets')
    parser.add_argument('--batch_size', type=int, default=32, 
                      help='Batch size for training and evaluation')
    parser.add_argument('--epochs', type=int, default=100, 
                      help='Epochs for trigger generation')
    parser.add_argument('--lr', type=float, default=0.01, 
                      help='Learning rate')
    
    # Loss component weights
    parser.add_argument('--lambda1', type=float, default=0.8, 
                      help='Weight for clean similarity loss')
    parser.add_argument('--lambda2', type=float, default=0.5, 
                      help='Weight for adversarial similarity loss')
    parser.add_argument('--lambda3', type=float, default=1.0, 
                      help='Weight for attack effectiveness loss')
    
    # Attack parameters
    parser.add_argument('--epsilon', type=float, default=8/255, 
                      help='Epsilon for PGD attack')
    parser.add_argument('--target_class', type=int, default=0, 
                      help='Target class for backdoor attack')
    
    # File paths
    parser.add_argument('--reference_trigger_path', type=str, default='optimized_trigger.pth', 
                      help='Path to save/load the reference trigger')
    parser.add_argument('--threshold_file_path', type=str, default='dynamic_threshold.txt', 
                      help='Path to save/load the dynamic threshold')
    parser.add_argument('--candidate_triggers_dir', type=str, default='./candidate_triggers', 
                      help='Directory with candidate triggers to check')
    parser.add_argument('--trigger_to_analyze', type=str, default=None,
                      help='Specific trigger file to analyze (for analyze_trigger mode)')
    
    # Additional options
    parser.add_argument('--debug', action='store_true', 
                      help='Enable debug visualizations')
    parser.add_argument('--seed', type=int, default=42, 
                      help='Random seed for reproducibility')
    parser.add_argument('--dataset', type=str, default='cifar10', 
                      choices=['cifar10', 'cifar100', 'imagenet'], 
                      help='Dataset to use')
    parser.add_argument('--robustness_factor_l1', type=float, default=1.0, 
                      help='Robustness factor for L1 norm dynamic threshold.')
    
    parser.add_argument('--eps_ref_backdoor_factor', type=float, default=0.7,
                        help='Factor to apply to reference backdoor EPS for setting threshold (e.g., 0.7 means 70% of ref EPS).')
    parser.add_argument('--eps_max_batches', type=int, default=10,
                        help='Max number of batches from calibration/test loader to use for EPS calculation (-1 for all).')

    # L1 Thresholding
    parser.add_argument('--l1_percentile_benign', type=float, default=95.0, help='Percentile for benign L1 norm distribution to set L1 upper threshold.')
    parser.add_argument('--l1_ref_factor_upper', type=float, default=1.5, help='Factor to multiply ref_trigger L1 to get an alternative L1 upper threshold.')
    # EPS Thresholding
    parser.add_argument('--eps_benign_nsigma', type=float, default=2.5, help='N-sigma for EPS benign threshold.') # Adjusted default
    parser.add_argument('--eps_factor_on_difference', type=float, default=0.3, help='Factor [0,1] to set EPS threshold between mean_benign_eps and ref_backdoor_eps.')
    # ASR Thresholding
    parser.add_argument('--asr_benign_nsigma', type=float, default=2.5, help='N-sigma for ASR benign threshold.') # Adjusted default
    parser.add_argument('--asr_factor_on_difference', type=float, default=0.3, help='Factor [0,1] to set ASR threshold between mean_benign_asr and ref_backdoor_asr.')
    # General
    parser.add_argument('--eps_asr_max_batches_calib', type=int, default=20, help='Max batches for EPS/ASR calculation during calibration.')
    parser.add_argument('--eps_asr_max_batches_check', type=int, default=10, help='Max batches for EPS/ASR calculation during trigger checking.')
    # target_class is already there, used for ASR
    parser.add_argument('--embedding_dim_detector', type=int, default=128, help='Embedding dim for SimpleEncoder if used in detector.')
    parser.add_argument('--num_classes_detector', type=int, default=10, help='Num classes for SimpleEncoder if used in detector.')




    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Define transforms and model (ensure this model is consistent)
    if args.dataset in ['cifar10', 'cifar100']:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        img_h, img_w = 32, 32
    else:  # imagenet
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img_h, img_w = 224, 224
    
    # Create model (this model instance will be used for EPS calculations)
    if args.dataset == 'cifar10':
        model = BackdoorModel(num_classes=10, feature_dim=128).to(device)
    elif args.dataset == 'cifar100':
        model = BackdoorModel(num_classes=100, feature_dim=128).to(device)
    else:  # imagenet
        model = BackdoorModel(num_classes=1000, feature_dim=256).to(device)
    # IMPORTANT: If your 'optimized_trigger.pth' was generated with a specific pretrained model
    # or a model in a specific state, you need to load that state into 'model' here
    # for the EPS calculations to be meaningful in 'generate_reference'.
    # For 'detect_backdoors', this 'model' is the one used to probe the candidate triggers.

    if args.mode == 'generate_reference':
        if args.dataset == 'cifar10':
            general_model = BackdoorModel(num_classes=10, feature_dim=128).to(device) # ResNet-based
        elif args.dataset == 'cifar100':
            general_model = BackdoorModel(num_classes=100, feature_dim=128).to(device)
        else: # imagenet
            general_model = BackdoorModel(num_classes=1000, feature_dim=256).to(device)
        # Potentially load a pretrained state for general_model if reference generation depends on it
        logging.info(f"Using BackdoorModel (ResNet-based) for '{args.mode}' mode.")
    else: # For detect_backdoors and analyze_trigger, the model might change per trigger
        # We'll define a 'default_clean_model' for checking benign files or as a fallback.
        # This default_clean_model should be SimpleEncoder if we are checking SimpleEncoder backdoors.
        default_clean_model = SimpleEncoder(num_classes=args.num_classes_detector, embedding_dim=args.embedding_dim_detector).to(device)
        # It's good practice to load clean pre-trained weights if available, or train it briefly on clean data.
        # For simplicity here, it's just instantiated.
        logging.info(f"Using SimpleEncoder as the base model architecture for '{args.mode}' mode when checking generated triggers.")
        general_model = default_clean_model # For detect_backdoors, this general_model acts as a fallback/benign checker

    
    
    
    # Mode-specific operations
    if args.mode == 'generate_reference':
        logging.info("Mode: generate_reference - Creating reference trigger and thresholds")
        
        # Load dataset for calibration (used for EPS calculation)
        if args.dataset == 'cifar10':
            train_dataset_for_opt = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform)
            calibration_dataset = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform)
        elif args.dataset == 'cifar100':
            train_dataset_for_opt = datasets.CIFAR100(root=args.data_dir, train=True, download=True, transform=transform)
            calibration_dataset = datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=transform)
        else: 
            # For ImageNet, provide a path to a subset for calibration
            # This example uses CIFAR10 as a placeholder for ImageNet data loading logic
            calibration_dataset = datasets.ImageFolder(root=os.path.join(args.data_dir, 'val_subset_for_calib'), transform=transform) # Example for ImageNet subset
            if not os.path.exists(os.path.join(args.data_dir, 'val_subset_for_calib')):
                 logging.warning(f"Calibration subset for ImageNet not found at {os.path.join(args.data_dir, 'val_subset_for_calib')}. Using CIFAR10 for calibration as placeholder.")
                 calibration_dataset = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform)

        # Use a subset for faster processing
        subset_size_opt = min(1000, len(train_dataset_for_opt))
        indices_opt = torch.randperm(len(train_dataset_for_opt))[:subset_size_opt]
        train_subset_for_opt = Subset(train_dataset_for_opt, indices_opt)
        opt_loader = DataLoader(train_subset_for_opt, batch_size=args.batch_size, shuffle=True)
        
        calibration_loader = DataLoader(calibration_dataset, batch_size=args.batch_size, shuffle=False)

        inputs_for_opt, _ = next(iter(opt_loader))
        inputs_for_opt = inputs_for_opt.to(device)
        # Infer img_shape from loaded data for multi_objective_optimization
        actual_img_shape = inputs_for_opt.shape[1:] 
        
        logging.info(f"Starting multi-objective optimization for reference trigger...")
        #logging.info(f"Input shape: {img_shape}, Target class: {args.target_class}")
        
        # Generate reference trigger
        optimized_trigger = multi_objective_optimization(
            general_model, inputs_for_opt, actual_img_shape,
            target_class=args.target_class, lambda1=args.lambda1, lambda2=args.lambda2, lambda3=args.lambda3,
            epochs=args.epochs, lr=args.lr, epsilon=args.epsilon, device=device
        )
        
        # Save optimized trigger
        torch.save(optimized_trigger, args.reference_trigger_path)
        logging.info(f"Optimized reference trigger saved to {args.reference_trigger_path}")

        # Calculate and save dynamic threshold
        logging.info(f"Generating dynamic thresholds using target_class: {args.target_class} for ASR calculations.")
        threshold_values = dynamic_threshold_adaptation(
            model, calibration_loader, optimized_trigger, args.target_class, device,
            l1_percentile_benign=args.l1_percentile_benign,
            l1_ref_factor_upper=args.l1_ref_factor_upper,
            eps_benign_nsigma=args.eps_benign_nsigma,
            eps_factor_on_difference=args.eps_factor_on_difference,
            asr_benign_nsigma=args.asr_benign_nsigma,
            asr_factor_on_difference=args.asr_factor_on_difference,
            eps_asr_max_batches_calib=args.eps_asr_max_batches_calib,
            min_asr_threshold=0.2, 
            min_eps_threshold=0.01 
        )
        with open(args.threshold_file_path, 'w') as f:
            json.dump(threshold_values, f, indent=4)
        logging.info(f"Dynamic thresholds saved to {args.threshold_file_path}: {threshold_values}")

        
    elif args.mode == 'detect_backdoors':
        logging.info("Mode: detect_backdoors - Checking candidate triggers for backdoors")
        
        # Load threshold
        if not os.path.exists(args.threshold_file_path):
            logging.error(f"Threshold file {args.threshold_file_path} not found. Generate it first.")
            return
            
        with open(args.threshold_file_path, 'r') as f:
            thresholds = json.load(f) # Load JSON
        logging.info(f"Loaded dynamic thresholds: {thresholds}")
        
        # Check for candidates directory
        if not os.path.isdir(args.candidate_triggers_dir):
            logging.error(f"Candidate triggers directory {args.candidate_triggers_dir} not found.")
            return
        
        # Need a DataLoader for clean test data for EPS calculation
        if args.dataset == 'cifar10':
            test_dataset = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform)
        elif args.dataset == 'cifar100':
            test_dataset = datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=transform)
        else: # imagenet placeholder
            logging.warning("Using CIFAR-10 as a placeholder for ImageNet test dataset loading. Adapt as needed.")
            test_dataset = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform) # Placeholder
        
        #test_loader_for_eps = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader_for_check = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        

        # Find all potential trigger files
        candidate_files_with_type = [] # Store (filepath, type_of_trigger)
        
        # Search in with_trigger for backdoored ones
        with_trigger_dir = os.path.join(args.candidate_triggers_dir, "with_trigger")
        if os.path.isdir(with_trigger_dir):
            for ext in ['*.pth']: # Only .pth for triggers saved by generate_test_triggers.py
                for f_path in glob.glob(os.path.join(with_trigger_dir, ext)):
                    if "backdoor_trigger_" in os.path.basename(f_path): # Ensure it's a trigger file
                         candidate_files_with_type.append((f_path, "with_trigger"))

        # Search in without_trigger for benign ones
        without_trigger_dir = os.path.join(args.candidate_triggers_dir, "without_trigger")
        if os.path.isdir(without_trigger_dir):
             for ext in ['*.pth']:
                for f_path in glob.glob(os.path.join(without_trigger_dir, ext)):
                     candidate_files_with_type.append((f_path, "without_trigger"))
        
        # Also search directly in candidate_triggers_dir for any other .pth files (e.g. optimized_trigger.pth)
        for ext in ['*.pth', '*.pt']:
            for f_path in glob.glob(os.path.join(args.candidate_triggers_dir, ext)):
                # Avoid double-adding if already found in subdirs
                is_already_added = False
                for added_f, _ in candidate_files_with_type:
                    if os.path.samefile(added_f, f_path):
                        is_already_added = True
                        break
                if not is_already_added:
                    candidate_files_with_type.append((f_path, "general_candidate"))


        if not candidate_files_with_type:
            logging.warning(f"No potential trigger files found in {args.candidate_triggers_dir} or its subdirectories 'with_trigger', 'without_trigger'.")
            return

        logging.info(f"Found {len(candidate_files_with_type)} candidate files to check.")
        results = []

        for trigger_file_path, trigger_type in tqdm(candidate_files_with_type, desc="Checking Triggers"):
            model_for_this_check = None
            
            if trigger_type == "with_trigger":
                # This is a trigger generated by generate_test_triggers.py
                # We need to load its corresponding backdoored SimpleEncoder model
                # and its specific target_label.
                trigger_filename = os.path.basename(trigger_file_path) # e.g., backdoor_trigger_001.pth
                model_id = trigger_filename.replace("backdoor_trigger_", "").replace(".pth", "")
                model_filename = f"backdoored_encoder_{model_id}.pth"
                model_path = os.path.join(with_trigger_dir, model_filename)

                if os.path.exists(model_path):
                    logging.info(f"  Loading specific backdoored SimpleEncoder: {model_path}")
                    # Ensure args.num_classes_detector and args.embedding_dim_detector match what SimpleEncoder was trained with
                    # In generate_test_triggers.py, num_classes=NUM_CLASSES (10 for CIFAR10), embedding_dim=args.embedding_dim (default 128)
                    current_model_instance = SimpleEncoder(num_classes=args.num_classes_detector, # Should match NUM_CLASSES from generator
                                                           embedding_dim=args.embedding_dim_detector).to(device) # Should match embedding_dim from generator
                    try:
                        current_model_instance.load_state_dict(torch.load(model_path, map_location=device))
                        model_for_this_check = current_model_instance
                    except Exception as e:
                        logging.error(f"    Failed to load state_dict for {model_path}: {e}. Using default clean model.")
                        model_for_this_check = default_clean_model # Fallback
                else:
                    logging.warning(f"  Could not find corresponding model {model_path} for trigger {trigger_file_path}. Using default clean model.")
                    model_for_this_check = default_clean_model # Fallback
            
            else: # "without_trigger" or "general_candidate"
                # For benign triggers or other candidates, use the default_clean_model (SimpleEncoder)
                # or if you want to check them against the ResNet based 'general_model' (if args.mode was 'generate_reference' with ResNet)
                # For consistency in detecting generate_test_triggers.py outputs, use SimpleEncoder here too.
                model_for_this_check = default_clean_model
                logging.info(f"  Using default clean SimpleEncoder for: {os.path.basename(trigger_file_path)}")


            is_backdoor, metrics_dict = check_trigger_backdoor(
                model_for_this_check, # Pass the model instance to use
                test_loader_for_check,
                trigger_file_path, # Path to the trigger .pth file
                thresholds,
                args.target_class, # Default target_class (will be overridden if trigger file has one)
                device,
                debug=args.debug,
                eps_asr_max_batches_check=args.eps_asr_max_batches_check
            )
            results.append({
                "file": os.path.basename(trigger_file_path),
                "type": trigger_type,
                "is_backdoor": is_backdoor,
                "l1_norm": metrics_dict.get('l1_norm', -1),
                "eps_score": metrics_dict.get('eps_score', -1),
                "asr_score": metrics_dict.get('asr_score', -1),
                "target_class_used": metrics_dict.get('final_target_class', args.target_class),
                "error": metrics_dict.get("error", None)
            })
        # ... (Summarize and save report as before, maybe include 'type' and 'target_class_used' in report)
        # Example modification for report:
        with open('backdoor_detection_report.txt', 'w') as f:
            f.write(f"Backdoor Detection Report\n")
            f.write(f"========================\n")
            f.write(f"Thresholds Used: {json.dumps(thresholds)}\n")
            f.write(f"Default Target Class for ASR (if not in trigger file): {args.target_class}\n\n")
            backdoor_count = sum(1 for res in results if res["is_backdoor"])
            f.write(f"Summary: {backdoor_count} backdoors found out of {len(results)} candidates.\n\n")
            f.write(f"{'File':<40} | {'Type':<15} | {'L1 Norm':<10} | {'EPS Score':<10} | {'ASR (Tgt)':<15} | {'Detected':<8}\n")
            f.write(f"{'-'*40} | {'-'*15} | {'-'*10} | {'-'*10} | {'-'*15} | {'-'*8}\n")

            for res in sorted(results, key=lambda x: (x.get("type", ""), x.get("file", ""))):
                if res["error"]:
                # 获取 target_class_used 并处理默认值及类型转换
                    target_class = res.get("target_class_used", args.target_class)
                    target_class_str = f"ERR ({int(target_class)})" if target_class is not None else "ERR (N/A)"
                    f.write(
                        f"{res['file']:<40} | {res.get('type','N/A'):<15} | {'ERROR':<10} | {'ERROR':<10} | "
                        f"{target_class_str:<15} | {False:<8} ({res['error']})\n"
                    )
            else:
                # 确保键值存在性（假设在无 error 时这些键一定存在）
                asr_display = f"{res['asr_score']:.4f} ({res['target_class_used']})"
                f.write(
                    f"{res['file']:<40} | {res.get('type','N/A'):<15} | {res['l1_norm']:.6f} | "
                    f"{res['eps_score']:.6f} | {asr_display:<15} | {str(res['is_backdoor']):<8}\n"
                )
        logging.info(f"Detailed report saved to backdoor_detection_report.txt")







    elif args.mode == 'analyze_trigger':
        logging.info("Mode: analyze_trigger - Analyzing a specific trigger file")

        
        if not args.trigger_to_analyze:
            logging.error("No trigger file specified for analysis. Use --trigger_to_analyze <path_to_file>")
            return
        if not os.path.exists(args.trigger_to_analyze):
            logging.error(f"Specified trigger file not found: {args.trigger_to_analyze}")
            return

        if not os.path.exists(args.threshold_file_path):
            logging.warning(f"Threshold file {args.threshold_file_path} not found. Analysis will proceed without thresholds.")
            thresholds_analyze = {'tau_l1': float('inf'), 'tau_eps': float('inf')} # Default to non-blocking thresholds
        else:
            with open(args.threshold_file_path, 'r') as f:
                thresholds_analyze = json.load(f)
        logging.info(f"Using thresholds for analysis: {thresholds_analyze}")

        if args.dataset == 'cifar10':
            test_dataset_analyze = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform)
        # ... (add other datasets for test_loader_analyze) ...
        else: # Fallback or specific dataset
            test_dataset_analyze = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform)
        
        test_loader_analyze = DataLoader(test_dataset_analyze, batch_size=args.batch_size, shuffle=False)

        model_to_analyze_with = SimpleEncoder(num_classes=args.num_classes_detector, embedding_dim=args.embedding_dim_detector).to(device)
        # Potentially load specific model if trigger_to_analyze has a linked model
        trigger_base = os.path.basename(args.trigger_to_analyze)
        if "backdoor_trigger_" in trigger_base:
            model_id = trigger_base.replace("backdoor_trigger_", "").replace(".pth", "")
            associated_model_path = os.path.join(os.path.dirname(args.trigger_to_analyze), f"backdoored_encoder_{model_id}.pth")
            if os.path.exists(associated_model_path):
                logging.info(f"Found associated model for analysis: {associated_model_path}. Loading it.")
                try:
                    model_to_analyze_with.load_state_dict(torch.load(associated_model_path, map_location=device))
                except Exception as e:
                    logging.error(f"Failed to load associated model {associated_model_path}: {e}. Using fresh SimpleEncoder.")
                    model_to_analyze_with = SimpleEncoder(num_classes=args.num_classes_detector, embedding_dim=args.embedding_dim_detector).to(device) # Re-instance
            else:
                logging.info(f"No associated backdoored model found for {trigger_base}. Analyzing with a fresh SimpleEncoder instance.")
        else:
            logging.info(f"Analyzing {trigger_base} with a fresh SimpleEncoder instance.")



        is_b, metrics_val = check_trigger_backdoor(
            model_to_analyze_with, # Pass the model to use
            test_loader_analyze, args.trigger_to_analyze,
            thresholds_analyze, args.target_class, device, debug=True,
            eps_asr_max_batches_check=args.eps_asr_max_batches_check
        )
        logging.info(f"  L1 Norm  : {metrics_val.get('l1_norm', -1):.6f} (Threshold: < {thresholds_analyze.get('tau_l1_upper', 'N/A'):.6f})")
        logging.info(f"  EPS Score: {metrics_val.get('eps_score', -1):.6f} (Threshold: > {thresholds_analyze.get('tau_eps_lower', 'N/A'):.6f})")
        logging.info(f"  ASR Score: {metrics_val.get('asr_score', -1):.6f} (Threshold: > {thresholds_analyze.get('tau_asr_lower', 'N/A'):.6f})")
        logging.info(f"  Detected as Backdoor: {is_b}")
        logging.info(f"  Target Class Used for ASR: {metrics_val.get('final_target_class', args.target_class)}")




    else: # Invalid mode
        logging.error(f"Invalid mode: {args.mode}. Choose from 'generate_reference', 'detect_backdoors', 'analyze_trigger'.")

if __name__ == "__main__":
    main()