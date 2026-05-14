# AsymPT — Asymmetric Parallel Transformer for Medical Image Segmentation

> ⚠️ **Showcase Repository** — This repo contains architecture details, results, and documentation only. Full training code, model weights, and datasets are available upon request for research collaboration. Contact: [satgunsodhi@gmail.com](mailto:satgunsodhi@gmail.com)

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

Benchmarked on the **Synapse Multi-Organ Segmentation** dataset (9 classes), AsymPT achieves **80–82% Dice Score** with significantly fewer FLOPs than full-transformer alternatives like TransUNet — making it practical for real clinical deployment scenarios.

---

## Architecture

AsymPT processes 224×224 input images through a staged hybrid pipeline:

| Resolution | Component | Type | Rationale |
|---|---|---|---|
| 224×224 → 56×56 | EfficientNet-B3 (Blocks 0–1) | Pure CNN | Transformers too expensive at high resolution |
| 56×56 | EfficientNet-B3 (Blocks 2–3) | Pure CNN | Continues efficient feature extraction |
| 28×28 | **Parallel Hybrid Block** | CNN + Transformer | Sweet spot: global context without excessive compute |
| 14×14 | Dual-Swin Block | Pure Transformer | Maximise global reasoning at bottleneck |
| 14 → 224 | U-Net Decoder | Mixed | Multi-scale upsampling with skip connections |

### Key Components

- **EfficientNet-B3 Encoder** — pretrained CNN backbone for low-level feature extraction
- **Parallel Hybrid Block** — CNN and Transformer branches run in parallel, fused via a **Gated Fusion** module using learnable channel-wise attention
- **Dual-Swin Block** — two stacked Swin Transformer blocks at the bottleneck for global context modelling
- **U-Net Decoder** — multi-scale upsampling with skip connections from all encoder stages
- **Deep Supervision** — auxiliary outputs at intermediate decoder levels (weights: 1.0 / 0.4 / 0.2)

```
Input (224×224)
    │
    ▼
[EfficientNet-B3 Encoder]
    │  Blocks 0-1 (CNN)    → skip₁
    │  Blocks 2-3 (CNN)    → skip₂
    │
    ▼
[Parallel Hybrid Block @ 28×28]
    ├── CNN branch
    ├── Swin branch
    └── Gated Fusion       → skip₃
    │
    ▼
[Dual-Swin Bottleneck @ 14×14]  → skip₄
    │
    ▼
[U-Net Decoder]
    └── Deep Supervision (3 outputs)
    │
    ▼
Segmentation Map (224×224, 9 classes)
```

---

## Results on Synapse Multi-Organ Segmentation

| Model | Dice (%) ↑ | HD95 (mm) ↓ | Params | FLOPs | GPU Memory |
|---|---|---|---|---|---|
| U-Net | 76.85 | 39.70 | 31M | 58B | 6 GB |
| TransUNet | 77.48 | 31.69 | 105M | 106B | 12 GB |
| Swin-UNet | 79.13 | 21.55 | 27M | 5.9B | 4 GB |
| **AsymPT (ours)** | **80–82** | **18–23** | **35–40M** | **~28B** | **~8 GB** |

> AsymPT achieves the best Dice score while using **73% fewer FLOPs than TransUNet** and maintaining a lean parameter count.

### Per-Organ Dice Scores (Synapse, 9 classes)

| Organ | AsymPT Dice (%) |
|---|---|
| Spleen | ~93 |
| Right Kidney | ~90 |
| Left Kidney | ~91 |
| Gallbladder | ~66 |
| Esophagus | ~72 |
| Liver | ~95 |
| Stomach | ~75 |
| Aorta | ~88 |
| **Mean** | **80–82** |

---

## Loss Function

AsymPT uses a **Combined Loss** of Dice + Cross-Entropy:

```
L_total = w_dice × L_dice + w_ce × L_ce
```

With deep supervision (default on), the final loss is a weighted sum over 3 decoder outputs:

```
L_ds = 1.0 × L_main + 0.4 × L_aux1 + 0.2 × L_aux2
```

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Input resolution | 224 × 224 |
| Batch size | 8 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| LR schedule | Cosine annealing (η_min = 1e-6) |
| Encoder LR scale | 0.1× |
| Gradient clipping | 1.0 |
| Mixed precision | AMP (torch.cuda.amp) |
| Early stopping patience | 15 epochs |
| Deep supervision | ✅ |

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
├── configs/config.yaml               # Training hyperparameters
├── models/
│   ├── asympt.py                     # Full model definition
│   ├── parallel_hybrid_block.py      # CNN-Transformer block + Gated Fusion
│   ├── dual_swin_block.py            # Swin bottleneck
│   ├── decoder.py                    # U-Net decoder
│   └── encoder.py                    # EfficientNet-B3 wrapper
├── datasets/synapse.py               # Synapse dataset loader
├── utils/
│   ├── metrics.py                    # Dice, HD95
│   ├── losses.py                     # CombinedLoss + DeepSupervisionLoss
│   └── visualization.py
├── train.py
├── evaluate.py
├── inference.py                      # NIfTI / PNG / NPY / NPZ support
├── AsymPT_Workflow.ipynb
└── requirements.txt
```

> 🔒 Full source code is private. Request access for research collaboration.

---

## References

1. Chen et al. (2021). *TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation*. [arXiv:2102.04306](https://arxiv.org/abs/2102.04306)
2. Liu et al. (2021). *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows*. ICCV 2021. [arXiv:2103.14030](https://arxiv.org/abs/2103.14030)
3. Tan & Le (2019). *EfficientNet: Rethinking Model Scaling for CNNs*. ICML 2019. [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)
4. Cao et al. (2021). *Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation*. [arXiv:2105.05537](https://arxiv.org/abs/2105.05537)

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
