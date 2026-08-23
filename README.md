# AsymPT — Asymmetric Parallel Transformer for Medical Image Segmentation

> ⚠️ **Showcase Repository** — This repo contains architecture overview, results, and documentation only. Full training code, model weights, and datasets are available upon request for research collaboration. Contact: [satgunsodhi@gmail.com](mailto:satgunsodhi@gmail.com)

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/framework-PyTorch-orange.svg)](https://pytorch.org/)
[![Status: Research](https://img.shields.io/badge/status-research-green.svg)]()

---

## Author

- **Satgun Singh Sodhi**

---

## Overview

**AsymPT** is a hybrid CNN-Transformer architecture for **medical image segmentation**, built around the principle of *Asymmetric Parallelism* — placing transformer components only where they provide the most benefit, and CNN components where efficiency matters most.

Benchmarked on the **Synapse Multi-Organ Segmentation** dataset (9 classes), AsymPT achieves state-of-the-art Dice scores with significantly fewer FLOPs than full-transformer alternatives like TransUNet — making it practical for real clinical deployment scenarios.

> **Latest stable run (2026-08):** Mean Dice **87.88%** on the Synapse validation set — trained for 219 epochs (best checkpoint at epoch 169) on a single NVIDIA Tesla T4 GPU, ~10.7 hours total.

---

## Architecture (High-Level)

AsymPT processes 224×224 input images through a staged hybrid pipeline:

| Resolution | Component | Type | Rationale |
|---|---|---|---|
| 224×224 → 56×56 | EfficientNet-B3 (early blocks) | Pure CNN | Transformers too expensive at high resolution |
| 56×56 | EfficientNet-B3 (mid blocks) | Pure CNN | Continues efficient feature extraction |
| 28×28 | **Parallel Hybrid Block** | CNN + Transformer | Sweet spot: global context without excessive compute |
| 14×14 | Dual-Swin Block | Pure Transformer | Maximise global reasoning at bottleneck |
| 14 → 224 | U-Net-style Decoder | Mixed | Multi-scale upsampling with skip connections |

### Key Components

- **EfficientNet-B3 Encoder** — pretrained CNN backbone for low-level feature extraction
- **Parallel Hybrid Block** — CNN and Transformer branches run in parallel, fused via a learnable **Gated Fusion** module
- **Dual-Swin Block** — stacked Swin Transformer blocks at the bottleneck for global context modelling
- **Multi-scale Decoder** — upsampling with skip connections and cross-stage feature fusion
- **Deep Supervision** — auxiliary outputs at intermediate decoder levels

```
Input (224×224)
    │
    ▼
[EfficientNet-B3 Encoder]  (staged CNN feature extraction)
    │
    ▼
[Parallel Hybrid Block @ 28×28]
    ├── CNN branch
    ├── Swin branch
    └── Gated Fusion
    │
    ▼
[Dual-Swin Bottleneck @ 14×14]
    │
    ▼
[Multi-scale Decoder]
    └── Deep Supervision (auxiliary outputs)
    │
    ▼
Segmentation Map (224×224, 9 classes)
```

---

## Results on Synapse Multi-Organ Segmentation

### Comparison with SOTA

| Model | Mean Dice (%) ↑ | HD95 (mm) ↓ | Params | FLOPs | GPU Memory |
|---|---|---|---|---|---|
| U-Net | 76.85 | 39.70 | 31M | 58B | 6 GB |
| TransUNet | 77.48 | 31.69 | 105M | 106B | 12 GB |
| Swin-UNet | 79.13 | 21.55 | 27M | 5.9B | 4 GB |
| **AsymPT (ours)** | **87.88** | — | **~35–40M** | **~28B** | **~8 GB** |

> AsymPT achieves a substantial Dice improvement over Swin-UNet and TransUNet while using **~73% fewer FLOPs than TransUNet** and a comparable, lean parameter count.

### Latest Stable Run — Per-Organ Dice (Synapse, 9 classes)

**Hardware:** single NVIDIA Tesla T4 GPU · **Training time:** ~10.7 hours (219 epochs, early stopping) · **Best checkpoint:** epoch 169

| Organ | AsymPT Dice (%) |
|---|---|
| Background | 99.45 |
| Spleen | 87.59 |
| Right Kidney | 92.02 |
| Left Kidney | 89.63 |
| Gallbladder | 88.90 |
| Esophagus | 88.07 |
| Liver | 83.93 |
| Stomach | 88.04 |
| Aorta | 84.87 |
| **Mean (excl. background)** | **87.88** |

> Earlier iterations of the model showed a large gap on small/hard organs (Gallbladder, Esophagus ~66–72% Dice). Subsequent improvements to the augmentation and loss strategy closed this gap substantially — both classes now score ~88%, in line with the larger organs.

---

## Training Setup (Summary)

| Aspect | Detail |
|---|---|
| Input resolution | 224 × 224 |
| Optimizer | AdamW with cosine-based LR scheduling |
| Loss | Combined region + boundary-aware loss with deep supervision |
| Regularization | Data augmentation pipeline (spatial + intensity), stochastic depth |
| Mixed precision | AMP |
| Hardware | Single NVIDIA Tesla T4 GPU |
| Training time | ~10.7 hours (219 epochs, early stopping) |

> Exact hyperparameters, loss formulation, and augmentation configuration are part of the private training pipeline — available on request for research collaboration.

---

## Tech Stack

`Python 3.10+` · `PyTorch` · `Swin Transformer` · `EfficientNet-B3` · `AMP` · `NIfTI` · `TensorBoard`

---

## Dataset

The model is evaluated on the **Synapse Multi-Organ CT Segmentation** dataset:

- 18 training cases / 12 test cases
- 9 abdominal organ classes
- Access: [Synapse Multi-Atlas Segmentation Challenge](https://www.synapse.org/#!Synapse:syn3193805/wiki/89480)

---

## Project Structure (Full Private Repo)

```
AsymPT/
├── configs/                          # Training hyperparameters
├── models/                           # Encoder, hybrid block, decoder definitions
├── datasets/                         # Synapse dataset loader
├── utils/                            # Metrics, losses, augmentation, visualization
├── train.py
├── evaluate.py
├── inference.py                      # NIfTI / PNG / NPY / NPZ support
├── AsymPT_Workflow.ipynb
└── requirements.txt
```

> 🔒 Full source code, exact module contents, and training configuration are private. Request access for research collaboration.

---

## References

1. Chen et al. (2021). *TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation*. [arXiv:2102.04306](https://arxiv.org/abs/2102.04306)
2. Liu et al. (2021). *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows*. ICCV 2021. [arXiv:2103.14030](https://arxiv.org/abs/2103.14030)
3. Tan & Le (2019). *EfficientNet: Rethinking Model Scaling for CNNs*. ICML 2019. [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)
4. Zhou et al. (2019). *UNet++: Redesigning Skip Connections to Exploit Multiscale Features*. IEEE TMI. [arXiv:1912.05074](https://arxiv.org/abs/1912.05074)
5. Cao et al. (2021). *Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation*. [arXiv:2105.05537](https://arxiv.org/abs/2105.05537)
6. Lin et al. (2017). *Focal Loss for Dense Object Detection*. ICCV 2017. [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)

---

## Citation

If you reference this work, please cite:

```bibtex
@misc{sodhi2026asympt,
  author       = {Satgun Singh Sodhi},
  title        = {AsymPT: Asymmetric Parallel Transformer for Medical Image Segmentation},
  year         = {2026},
  howpublished = {\url{https://github.com/satgunsodhi/AsymPT-showcase}},
  note         = {Research showcase. Full code available upon request.}
}
```

---

## Contact & Collaboration

For research collaboration, code access, or dataset sharing:

📧 [satgunsodhi@gmail.com](mailto:satgunsodhi@gmail.com)
🔗 [LinkedIn](https://linkedin.com/in/satgunsodhi)
🐙 [GitHub](https://github.com/satgunsodhi)
