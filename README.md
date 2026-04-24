# The Self-Pruning Neural Network 🧠✂️

**Tredence Analytics — AI Agents Engineering Internship 2025 Submission**

This repository contains the solution for the "Self-Pruning Neural Network" challenge. The goal of this project is to design and implement a neural network that dynamically learns to prune its own unnecessary weights *during* the training process, rather than as a post-training compression step.

## 🌟 Key Features

- **Custom `PrunableLinear` Layer:** A PyTorch `nn.Module` implementation where every weight is multiplied by a learnable gate ($\in (0,1)$).
- **Self-Balancing Sparsity Loss:** Utilizes an L1 penalty on the sigmoid gates. We specifically utilize the **sum** of the gates to ensure the mathematical gradient at the per-parameter level provides a strong, uniform downward pruning pressure.
- **Dynamic Routing:** As the gates approach $0$, the corresponding weights are effectively "pruned" on the fly.
- **Comprehensive Evaluation:** Evaluated on the CIFAR-10 dataset with a full analysis of the trade-off between the sparsity penalty ($\lambda$) and test accuracy.

## 📁 Repository Structure

- `self_pruning_nn.ipynb`: The primary, self-contained Jupyter Notebook. Contains the complete implementation of the Prunable layer, the CNN architecture, the exact sparsity loss, the training loop, and all visualization code. *(Optimized for execution in Kaggle / Google Colab).*
- `Report.md`: A detailed Markdown report explaining the mathematical foundation of the L1 penalty, analyzing the critical difference between `.sum()` and `.mean()` on the parameter gradients, and presenting the final accuracy-vs-sparsity results.
- `generate_notebook.py`: A Python script that programmatically generated the final IPYNB structure.
- `self_pruning_network.py`: A standalone Python script version of the codebase.

## 🚀 How to Run

1. **Open in Kaggle or Colab:**
   Upload `self_pruning_nn.ipynb` to a Kaggle Notebook or Google Colab environment.
2. **Select Accelerator:**
   *Important:* Ensure your hardware accelerator is set to a modern GPU (e.g., **GPU T4 x2** in Kaggle). Older GPUs like the P100 (`sm_60`) have been deprecated by recent PyTorch builds and will throw a `CUDA error: no kernel image is available`.
3. **Run All Cells:**
   The notebook will automatically download the CIFAR-10 dataset, train the network across three different $\lambda$ configurations (`0.0`, `0.01`, `0.05`), and generate the final plots.

## 📊 Results Summary

The L1 penalty successfully induces a bimodal gate distribution, allowing the network to aggressively drop parameters while maintaining competitive accuracy:

| Lambda ($\lambda$) | Test Accuracy | Sparsity Level |
|:---|:---:|:---:|
| **0.00** | ~85.5% | 0.00% |
| **0.01** | ~84.0% | ~40.0% |
| **0.05** | ~79.0% | ~85.0% |

For a deeper dive into the gradient mechanics and visual heatmaps, see [Report.md](./Report.md).

---
*Built with PyTorch. Designed for the Tredence Studio AI Engineering Cohort.*
