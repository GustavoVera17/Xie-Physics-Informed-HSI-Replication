# Dual Camera Snapshot HSI via Physics-Informed Learning (Replication)

This repository contains a PyTorch replication of the methodology proposed in the paper:
> **"Dual camera snapshot hyperspectral imaging system via physics-informed learning"** by Hui Xie et al. (2022).

## 🧠 Project Overview
Unlike traditional supervised learning that relies on ground truth Hyperspectral Images (HSIs) to calculate the L1 Loss, this implementation utilizes a **Self-Supervised Physics-Informed** approach. The network (based on HRNet-W18) acts as an untrained generator, and the loss is calculated by mathematically simulating the forward models of the optical hardware:
1. **SSL-Grayscale Branch:** Simulates the CASSI coded aperture and dispersive prism.
2. **SSL-Color Branch:** Simulates the color camera's spectral quantum efficiency curve (CRF).

## 📂 Repository Structure
* `dataset_dual.py`: Custom dataloader for CAVE dataset, simulating the hardware beam-splitter on-the-fly.
* `physics_loss.py`: Contains the `SSL_Grayscale_Loss` and `SSL_Color_Loss` simulated forward models.
* `sir_cnn.py`: The Spectral Image Reconstruction CNN (HRNet-W18 topology).
* `train_xie.py`: The closed-loop training script with checkpointing and auto-dashboard generation.

## 🚀 Status
Currently training and validating the physics-informed constraint mechanism.