Adversarial Robustness Analysis on CIFAR-10 using ResNet-18
Project Overview

This project analyzes the adversarial robustness of deep neural networks on the CIFAR-10 dataset using a modified ResNet-18 architecture. The study evaluates multiple adversarial training strategies under consistent experimental settings and examines the trade-off between clean accuracy and robustness against adversarial attacks.

The project was conducted as part of an academic study and focuses on L∞-bounded adversarial attacks, including FGSM, PGD, and TRADES-based defenses.

Authors

Murat Çevlik – 152120201067

Onur Dalgıç – 15212021068

Objectives

Implement a ResNet-18 architecture adapted for CIFAR-10

Compare multiple adversarial training strategies

Evaluate robustness under increasing adversarial perturbation budgets

Analyze fine-tuning and curriculum-based adversarial training methods

Examine the robustness–accuracy trade-off across different models

Dataset

CIFAR-10

60,000 RGB images (32×32)

10 classes

Preprocessing

Normalization using CIFAR-10 mean and standard deviation

Random crop (padding = 4)

Random horizontal flip (p = 0.5)

Model Architecture

Modified ResNet-18

3×3 convolution (stride 1) instead of ImageNet-style 7×7

Batch Normalization

Global Average Pooling

Architecture adapted for small-resolution images

Threat Model & Attacks

All attacks operate under an L∞ constraint.

FGSM (Fast Gradient Sign Method)

PGD (Projected Gradient Descent)

PGD-7 for adversarial training

PGD-10 for fine-tuning

TRADES Loss

β = 6.0

Evaluation performed at:

ε = 2/255

ε = 4/255

ε = 8/255

Training Strategies

Eight different models were trained:

Model	Strategy
M1	Standard (Clean Training)
M2	Pure FGSM (ε = 8/255)
M3	Hybrid Clean + FGSM
M4	FGSM Mixed (ε = 4/255)
M5	PGD-7 Adversarial Training
M6	TRADES Training
M7	PGD-10 Fine-Tuned + Label Smoothing
M8	Curriculum PGD (WideResNet-28-10)
Results Summary

Clean-only training achieves high accuracy but fails under adversarial attacks

FGSM-based defenses improve robustness at low ε but degrade under stronger attacks

PGD and TRADES provide stable robustness with reduced clean accuracy

Fine-tuning PGD with stronger attacks and label smoothing improves robustness

Curriculum PGD with WideResNet-28-10 achieved the best overall performance

Best Robust Accuracy

60.42% at ε = 8/255 (Curriculum PGD – WideResNet-28-10)

Key Findings

Adversarial robustness strongly depends on both training strategy and model capacity

Curriculum-based training significantly improves stability and performance

Wider architectures provide a better robustness–accuracy trade-off

Fine-tuning can recover clean accuracy while maintaining robustness

Tools & Frameworks

PyTorch

CUDA-enabled GPU training

Standard adversarial training implementations

Known Limitations

High computational cost for PGD-based methods

Increased memory usage for wide architectures

Robustness evaluation limited to first-order attacks

Future Work

Architecture-controlled comparisons

Stronger and adaptive attack evaluations

AutoAttack-based robustness benchmarks

Cross-dataset robustness analysis

