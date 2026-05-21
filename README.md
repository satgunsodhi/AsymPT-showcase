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

An optional **Focal Loss** term can be added for hard-example mining, down-weighting easy examples and focusing training on small, hard-to-segment organ classes (e.g. Gallbladder, Esophagus):

```
L_total = w_dice × L_dice + w_ce × L_ce + w_focal × L_focal
```

With deep supervision (default on), the final loss is a weighted sum over 3 decoder outputs:

```
L_ds = 1.0 × L_main + 0.4 × L_aux1 + 0.2 × L_aux2
```

`CombinedLoss` also supports a `label_smoothing` parameter to reduce overconfident predictions.

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
| Label smoothing | Configurable (`CombinedLoss`) |
| EMA | ✅ Exponential Moving Average model wrapper |

---

## Data Augmentation Pipeline

AsymPT uses an augmentation pipeline specifically designed to address the overfitting gap (train Dice ~96% vs val Dice ~88%) observed in abdominal CT segmentation:

| Augmentation | Type | Details |
|---|---|---|
| **Elastic Deformation** | Spatial | Simulates soft tissue motion; smooth random displacement field (alpha=50, sigma=5, p=0.3) |
| **Random Scale + Crop** | Spatial | Scale range 0.8–1.2×, crop back to 224×224; multi-scale organ robustness (p=0.5) |
| **Horizontal / Vertical Flip** | Spatial | Standard flips (p=0.5 each) |
| **Random 90° Rotation** | Spatial | 0/90/180/270° rotation |
| **Brightness / Contrast Jitter** | Intensity | Scale ±10%, shift ±0.1 (p=0.5) |
| **Gaussian Noise** | Intensity | σ ∈ [0.02, 0.08] (p=0.5) |
| **Gaussian Blur** | Intensity | kernel=5, σ ∈ [0.5, 1.5] (p=0.3) |
| **Cutout / Random Erasing** | Occlusion | 1 hole, 5–15% of image area (p=0.3); image only, label unchanged |

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
├── configs/
│   └── config.yaml                   # Training hyperparameters
├── models/
│   ├── __init__.py
│   ├── asympt.py                     # Full model definition + get_parameter_groups()
│   ├── parallel_hybrid_block.py      # Core CNN-Transformer parallel block + Gated Fusion
│   ├── dual_swin_block.py            # Swin Transformer bottleneck (2 blocks)
│   ├── decoder.py                    # U-Net style multi-scale decoder
│   └── encoder.py                    # EfficientNet-B3 encoder wrapper
├── datasets/
│   ├── __init__.py
│   └── synapse.py                    # Synapse Multi-Organ dataset + SynapseDataModule
├── utils/
│   ├── __init__.py
│   ├── metrics.py                    # Dice, HD95 (per-class + mean)
│   ├── losses.py                     # CombinedLoss (Dice + CE + Focal) + DeepSupervisionLoss
│   ├── visualization.py             # Segmentation overlay, attention maps, predictions grid
│   ├── augmentation.py              # Elastic deformation, scale/crop, cutout pipeline
│   └── ema.py                        # Exponential Moving Average model wrapper
├── train.py                          # Full training pipeline with AMP, early stopping, TensorBoard
├── evaluate.py                       # Evaluation with per-class and per-sample metrics
├── inference.py                      # Inference on NIfTI, PNG, NPY, NPZ images
├── test_model.py                     # Model unit tests
├── AsymPT_Workflow.ipynb             # End-to-end walkthrough notebook
├── Demo_Segmentation.ipynb           # Segmentation demo notebook
├── requirements.txt
├── setup.py
└── README.md
```

> 🔒 Full source code is private. Request access for research collaboration.

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
