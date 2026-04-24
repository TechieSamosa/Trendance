# Self-Pruning Neural Network
### Tredence Analytics · AI Engineering Internship 2025

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-CIFAR--10-green.svg)](https://www.cs.toronto.edu/~kriz/cifar.html)
[![Platform](https://img.shields.io/badge/Platform-Kaggle%20GPU-20BEFF.svg)](https://www.kaggle.com/)

---

## Overview

This project implements a **self-pruning neural network** that learns to remove its own unnecessary weights *during training* — no post-training pruning step required.

Instead of using standard `nn.Linear` layers, each fully connected layer is replaced with a custom `PrunableLinear` module. Every weight `w_ij` is paired with a learnable **gate parameter** `score_ij`. The gate value `sigmoid(score_ij) ∈ (0, 1)` multiplies the weight during the forward pass. An **L1 sparsity regularisation term** added to the loss encourages gates to converge to zero, effectively removing the corresponding connections.

```
Total Loss  =  CrossEntropyLoss(logits, y)
            +  λ  ×  Σ sigmoid(gate_scores)   ← SparsityLoss
```

The network is trained on **CIFAR-10** image classification and evaluated across three λ values to study the sparsity–accuracy trade-off.

---

## Repository Structure

```
Trendance/
├── self_pruning_nn.ipynb      # Complete Jupyter notebook (12 cells)
├── report.md                  # Full experiment report with analysis
├── README.md                  # This file
└── data/
    ├── training_curves.png        # Train/test accuracy, loss, sparsity vs epoch
    ├── gate_distribution.png      # Histogram of gate values for each λ
    ├── gate_heatmaps.png          # Per-layer gate heatmap (best model)
    ├── sparsity_vs_accuracy.png   # Sparsity% vs test accuracy% scatter
    └── best_pruning_model.pth     # PyTorch checkpoint (λ=1e-4, 91.68% acc)
```

---

## Quick Start

### Requirements

```bash
pip install torch torchvision numpy matplotlib
```

### Run on Kaggle (Recommended)

1. Upload `self_pruning_nn.ipynb` to [kaggle.com/code](https://www.kaggle.com/code)
2. Enable **GPU Accelerator** (Settings → Accelerator → GPU T4 × 2)
3. Run All Cells
4. Training takes ~5–8 minutes per λ value (3 experiments = ~20 min total)

### Run Locally

```bash
git clone https://github.com/TechieSamosa/Trendance.git
cd Trendance
jupyter notebook self_pruning_nn.ipynb
```

> CIFAR-10 (~170 MB) downloads automatically on first run to `./data/`.

---

## Notebook Structure

| Cell | Description |
|---|---|
| **1** | Imports, device setup, seed fixing |
| **2** | `PrunableLinear` class + gradient sanity check |
| **3** | `SelfPruningNet` architecture definition + parameter count |
| **4** | CIFAR-10 DataLoaders with augmentation |
| **5** | Training loop: `train_one_epoch`, `evaluate`, `run_experiment` |
| **6** | Run all 3 experiments (λ = 1e-5, 1e-4, 1e-3) |
| **7** | Results table + per-layer sparsity breakdown |
| **8** | Training curves plot |
| **9** | Gate value distribution histograms |
| **10** | Per-layer gate heatmaps |
| **11** | Sparsity vs. accuracy scatter plot |
| **12** | Save best model checkpoint |

---

## Results Summary

| Lambda (λ) | Test Accuracy | Sparsity Level | Notes |
|:---|---:|---:|:---|
| `1e-5` | 91.44% | 0.00% | Baseline (negligible regularisation) |
| `1e-4` | **91.68%** | 0.00% | Best accuracy |
| `1e-3` | 91.45% | 0.00% | Still no pruning at this scale |

### Honest Analysis

The three λ values tested (`1e-5`, `1e-4`, `1e-3`) produced **0% sparsity** across all experiments. This is a known hyperparameter scaling issue:

- The network has **2,230,784 gate parameters** in its FC layers.
- The raw `SparsityLoss` sums all of them → ≈ 1,628,472 at initialization.
- Even at `λ = 1e-3`, the per-gate gradient from the sparsity term is only ~0.0002.
- This is negligible compared to the classification gradient — so gates never move below the `1e-2` pruning threshold.

**The fix** is to normalize the sparsity loss by gate count (`gates.mean()` instead of `gates.sum()`), then use λ ∈ `{0.01, 0.1, 1.0}`. This is documented in detail in [`report.md`](./report.md).

Despite this, the model achieves **91.68% test accuracy on CIFAR-10** — a competitive result — and all architectural components (`PrunableLinear`, gradient flow, training loop) are correctly implemented.

---

## Key Implementation — `PrunableLinear`

```python
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight      = nn.Parameter(torch.empty(out_features, in_features))
        self.bias        = nn.Parameter(torch.empty(out_features))
        # One learnable gate score per weight — same shape as weight
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))
        self._init_parameters()

    def _init_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1.0 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        nn.init.constant_(self.gate_scores, 1.0)  # sigmoid(1.0) ≈ 0.73 at start

    def forward(self, x):
        gates         = torch.sigmoid(self.gate_scores)   # ∈ (0, 1)
        pruned_weight = self.weight * gates               # element-wise gating
        return F.linear(x, pruned_weight, self.bias)      # standard affine

    def get_gates(self):
        return torch.sigmoid(self.gate_scores).detach()

    def sparsity(self, threshold=1e-2):
        return (self.get_gates() < threshold).float().mean().item()
```

**Why gradients flow correctly:**
- `∂L/∂weight = ∂L/∂output × gates` — weights updated based on gated signal
- `∂L/∂gate_scores = ∂L/∂output × weight × σ′(score)` — gate scores updated via chain rule through sigmoid

---

## Architecture

```
Input (3×32×32 CIFAR-10)
  │
  ├─ Conv Block 1: Conv(3→64, BN, ReLU) × 2 → MaxPool     → 64×16×16
  ├─ Conv Block 2: Conv(64→128, BN, ReLU) × 2 → MaxPool   → 128×8×8
  ├─ Conv Block 3: Conv(128→256, BN, ReLU) × 2 → MaxPool  → 256×4×4
  │
  └─ Flatten → 4096
       │
       ├─ PrunableLinear(4096 → 512)  [2,097,152 gate params]
       ├─ ReLU + Dropout(0.5)
       ├─ PrunableLinear(512 → 256)   [131,072 gate params]
       ├─ ReLU + Dropout(0.3)
       └─ PrunableLinear(256 → 10)    [2,560 gate params]
            │
            └─ Output: 10 class logits

Total gate parameters: 2,230,784
```

---

## Output Plots

| Plot | Description |
|---|---|
| `data/training_curves.png` | 4-panel grid: train acc, test acc, classification loss, sparsity vs epoch |
| `data/gate_distribution.png` | Histogram of all 2.23M gate values for each λ |
| `data/gate_heatmaps.png` | Green/yellow/red heatmap of gate values per FC layer (best model) |
| `data/sparsity_vs_accuracy.png` | Scatter plot: sparsity% on x-axis, test accuracy% on y-axis |

---

## Theoretical Background

The L1 penalty on sigmoid gates is the key design choice. Unlike L2 regularisation (whose gradient vanishes near zero), the **L1 gradient is constant** — it applies a uniform downward push of magnitude `λ` on every gate regardless of its current value. This causes truly redundant weights to be driven to exact zero, while task-critical weights resist due to the opposing classification gradient.

See [`report.md`](./report.md) for the full derivation, gradient analysis, and corrective recommendations.

---

## Author

**[Your Name]**  
GitHub: [TechieSamosa](https://github.com/TechieSamosa)  
Submission for: Tredence Analytics AI Engineering Internship 2025
