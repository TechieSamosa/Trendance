# The Self-Pruning Neural Network — Experiment Report

## 1. Why Does an L1 Penalty on Sigmoid Gates Encourage Sparsity?

### The Gate Mechanism

Each weight $w_{ij}$ in a `PrunableLinear` layer is masked by a learnable gate:

$$\text{gate}_{ij} = \sigma(\text{score}_{ij}) \in (0, 1)$$
$$\text{pruned\_weight}_{ij} = w_{ij} \times \text{gate}_{ij}$$

where $\text{score}_{ij}$ is an `nn.Parameter` updated by the optimizer alongside the weight.

### The Total Loss

$$\text{Total Loss} = L_{CE}(\text{logits}, y) + \lambda \cdot \sum_{\text{layers}} \sum_{i,j} \sigma(\text{score}_{ij})$$

### Why L1 and Not L2?

| Regulariser | Gradient when gate $g \approx 0$ | Drives to exactly zero? |
|:---:|:---:|:---:|
| **L2** ($\sum g^2$) | $2g \to 0$ — vanishes as gate shrinks | ✗ Only asymptotically |
| **L1** ($\sum \|g\|$) | Constant $1$ regardless of magnitude | ✓ Uniform pressure |

The L1 norm applies a **constant downward force** on every gate's gradient. For each gate, there is a tug-of-war:
- If the weight **contributes to classification**, the task gradient wins → gate stays open
- If the weight is **redundant**, the sparsity gradient wins → gate is driven to zero

This per-weight competition is the self-pruning mechanism.

### The Sigmoid Gradient Challenge

While the L1 penalty provides constant pressure, the sigmoid's derivative $\sigma'(s) = \sigma(s)(1-\sigma(s))$ **vanishes** as $s \to -\infty$. This means that as a gate approaches zero, the gradient pushing it further down also shrinks. To overcome this, we use a **higher learning rate** for the gate parameters (0.05 vs 0.001 for weights), amplifying the sparsity signal enough to push gates past the pruning threshold.

---

## 2. Experimental Setup

| Hyperparameter | Value |
|---|---|
| Architecture | MLP: 3072 → 512 → 256 → 10 |
| Dataset | CIFAR-10 (50,000 train / 10,000 test) |
| Optimizer (weights) | Adam, lr = 1e-3 |
| Optimizer (gates) | Adam, lr = 0.05 |
| Epochs | 20 |
| Gate Initialization | Constant 1.0 → $\sigma(1.0) \approx 0.73$ |
| Sparsity Threshold | $10^{-2}$ |
| λ Values Tested | 1e-5, 1e-4, 5e-4, 1e-3, 2e-3 |
| Platform | Kaggle (GPU T4) |

---

## 3. Results

### 3.1 Summary Table

| Lambda (λ) | Test Accuracy (%) | Sparsity Level (%) |
|:---:|:---:|:---:|
| **1e-05** | **54.29** | **99.04** |
| 1e-04 | 47.37 | 99.83 |
| 5e-04 | 41.10 | 99.93 |
| 1e-03 | 33.43 | 99.97 |
| 2e-03 | 10.00 | 100.00 |

### 3.2 Analysis

**The self-pruning mechanism is highly effective.** Even at the smallest penalty ($\lambda = 10^{-5}$), the network prunes over 99% of its connections while retaining 54% test accuracy — a remarkable compression ratio.

**The accuracy-sparsity trade-off is clear and monotonic:**
- $\lambda = 10^{-5}$: Best balance. The network retains ~1% of its most critical connections, achieving reasonable accuracy on a hard task (CIFAR-10) with a simple MLP.
- $\lambda = 10^{-4}$ to $10^{-3}$: Progressively more aggressive pruning. Accuracy degrades as essential connections are also cut.
- $\lambda = 2 \times 10^{-3}$: Complete collapse. All gates are driven to zero (100% sparsity), and the model outputs random predictions (10% = chance level for 10 classes).

### 3.3 Gate Distribution

The gate value histograms confirm the expected **bimodal distribution**:
- A dominant spike at **0.0** — the pruned weights
- A small cluster near **0.5–1.0** — the surviving active connections

As λ increases, the spike at 0 grows and the surviving cluster shrinks, until at $\lambda = 2 \times 10^{-3}$ no gates survive.

### 3.4 Training Dynamics

The sparsity curves show a characteristic **sigmoidal growth pattern**:
1. **Early epochs (1–5):** Minimal pruning. The network prioritizes feature learning.
2. **Mid epochs (5–12):** Rapid pruning as the classification loss stabilizes and the L1 pressure takes over.
3. **Late epochs (12–20):** Saturation. The network reaches a stable sparse equilibrium.

This confirms the core hypothesis: **the model first learns, then compresses itself.**

---

## 4. Conclusion

1. **The `PrunableLinear` layer correctly implements the gated weight mechanism.** Gradients flow through both `weight` and `gate_scores` parameters, verified by explicit gradient checks.

2. **The L1 sparsity loss successfully drives gates to zero**, creating a genuinely sparse network during training.

3. **The λ hyperparameter provides a clear, controllable dial** for the sparsity-accuracy trade-off.

4. **A key implementation insight:** The gate parameters require a higher learning rate than the weights to overcome the sigmoid's vanishing gradient near zero. This is a practical engineering detail that makes the difference between 0% and 99% sparsity.
