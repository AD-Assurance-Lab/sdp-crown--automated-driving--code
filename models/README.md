# Trained E2E CARLA Driving Models

This directory contains the neural network weights for the end-to-end steering models trained on CARLA Town04, along with documentation of active checkpoints. All classification (MNIST/CIFAR) and reference (Udacity) models have been archived.

---

## 1. Active Checkpoints

We maintain weights for two model architectures:
*   **CarlaSteeringNet (`small`)**: Lightweight, verifier-friendly architecture. Features 4 Convolutional layers (no BatchNorm) followed by 3 Fully-Connected layers (no Dropout). It is designed to be highly verifiable without bounds explosion under CROWN and SDP-CROWN. It is trained using DAgger-Lite and weather augmentation on $120 \times 90$ image inputs.
*   **CarlaSteeringExpertNet (`expert`)**: High-capacity architecture. Features 4 Convolutional layers with Batch Normalization followed by 3 Fully-Connected layers with Dropout. It achieves excellent closed-loop driving stability but is computationally heavy for verification. To verify, BatchNorm layers must be fused into preceding Conv layers at eval/verification time.

The active weights files in this folder are:

| Filename | Architecture | Training Paradigm | Training Dataset | Purpose / Notes |
| :--- | :--- | :--- | :--- | :--- |
| `carla_expert_clear.pth` | `CarlaSteeringExpertNet` | DAgger-Lite | Clear Weather | Baseline expert model |
| `carla_expert_mixed.pth` | `CarlaSteeringExpertNet` | DAgger-Lite | Mixed Weather | Multi-weather robust expert model |
| `carla_small_clear.pth` | `CarlaSteeringNet` | DAgger-Lite | Clear Weather | Verifier-friendly clear baseline model |
| `carla_small_mixed.pth` | `CarlaSteeringNet` | DAgger-Lite | Mixed Weather | Verifier-friendly mixed baseline model |
| `carla_small_mixed_aug.pth` | `CarlaSteeringNet` | DAgger-Lite + Weather Aug | Clear + Weather Aug | Augmented mixed model to solve straight bias |

---

## 2. Model Naming Convention & Registry
To maintain diligent record-keeping, all trained CARLA models saved in `models/` must follow this naming scheme:
*   `carla_[architecture]_[weather]_[suffix].pth`
*   Where `[architecture]` is either `small` (lightweight `CarlaSteeringNet`) or `expert` (high-capacity `CarlaSteeringExpertNet`).
*   Where `[weather]` is either `clear` (trained on clear weather only) or `mixed` (trained on mixed-weather conditions).
*   Where `[suffix]` is optional (e.g., `aug` for weather augmentation, `kd` for knowledge distillation).
*   *Example:* `carla_small_clear.pth`, `carla_small_mixed_aug.pth`.
*   All old/deprecated models (MNIST, CIFAR, Udacity) must reside in `models/archive/`.

---

## 3. Evaluation and Verification Usage

### Closed-Loop Driving Simulator Evaluation:
To evaluate model steering stability in CARLA:
```bash
./venv_sdp/bin/python tools/test_carla_model.py \
    --model-type CarlaSteeringNet \
    --model-path models/carla_small_mixed_aug.pth \
    --weather rain \
    --num-frames 200 \
    --start-frame 750
```

### Offline Formal Bounds Verification:
To verify mathematical safety bounds under parameterized ACDC weather perturbations:
```bash
./venv_sdp/bin/python verify_steering.py \
    --model_type CarlaSteeringNet \
    --weights_path models/carla_small_mixed_aug.pth \
    --weather rain \
    --method SDP-CROWN \
    --device cuda \
    --num_frames 50
```
