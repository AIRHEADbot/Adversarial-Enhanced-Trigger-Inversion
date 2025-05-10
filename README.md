# Adversarially Robust Models for Enhanced Backdoor Detection

## Overview
This repository contains the implementation for our study focusing on two primary objectives:
1. **Experimental validation** of Projected Gradient Descent (PGD) adversarial training in improving neural networks' intrinsic stability against adversarial attacks.
2. **Demonstration** of leveraging stability-enhanced models for advanced backdoor detection using an Adversarially Enhanced Trigger Inversion method.

**Core Hypothesis**: Models trained with adversarial robustness exhibit more stable behavior—both in classification accuracy under attack and internal feature representation consistency. This stability enhances higher-level security analyses like backdoor trigger inversion.

---

## Dataset Preparation
- **Dataset**: CIFAR-10 
- **Path**: `./data/cifar10`

---

## Getting Started

### 1. Generate Test Cases
#### Generate Trigger Datasets
- `generate_test_triggers.py`: Creates test triggers including:
  - *Backdoored triggers* (trained via a poisoned SimpleEncoder model).
  - *Benign triggers* (random noise, zero tensors, or weights extracted from clean models).

#### Train Models
- **Control Group (Standard Training)**:
  ```bash
  python train.py --train_mode standard --num_models 5 --output_dir ./evaluation_models --epochs_per_encoder 20
  ```

- **Experimental Group (PGD Adversarial Training)**:
  ```bash
  python train.py --train_mode pgd --num_models 5 --output_dir ./evaluation_models --epsilon_pgd 0.03 --steps_pgd 10 --step_size_pgd 0.01 --epochs_per_encoder 20
  ```

Saved model weights will be stored in `./evaluation_models`.

---

### 2. Evaluate Model Stability
Run the evaluation script:
```bash
python run_stability_evaluation.py
```

**Note**: Due to computational constraints, the default setup uses a limited number of test samples. For higher statistical reliability, consider increasing the sample size in a more powerful environment.

---

## Results
![Experimental Results](./results.png)  
*(Replace with your actual results figure path)*

**Key Findings**:
- PGD adversarial training significantly improves model robustness against adversarial attacks in the DECREE framework.
- Enhanced stability in feature embeddings enables more reliable inversion and detection of backdoor triggers.

---

## Conclusion
Our experiments demonstrate that:
- PGD adversarial training effectively boosts encoder robustness in DECREE-based scenarios.
- Stability-enhanced models provide a stronger foundation for security-critical tasks like backdoor detection.

---

## References
- [DECREE Repository](https://github.com/GiantSeaweed/DECREE/tree/master)
- [BadEncoder Framework](https://github.com/jinyuan-jia/BadEncoder)
``` 

This README provides a structured, concise overview of your project. Replace `results.png` with your actual figure path, and expand sections like "Results" with specific metrics/plots as needed.
