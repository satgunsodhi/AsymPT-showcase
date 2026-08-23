# Model Card — AsymPT

## Model Description

- **Model name:** AsymPT (Asymmetric Parallel Transformer)
- **Task:** Multi-organ medical image segmentation
- **Framework:** PyTorch
- **Author:** Satgun Singh Sodhi

## Intended Use

- **Primary use:** Research in efficient medical image segmentation
- **Intended users:** Medical imaging researchers, computer vision practitioners
- **Out-of-scope:** Clinical diagnosis without human expert review; non-CT modalities (not validated)

## Inputs / Outputs

| | Detail |
|---|---|
| Input | CT scan slice or volume (NIfTI, PNG, NPY, NPZ) — normalized to 224×224 |
| Output | Per-pixel class labels for 9 abdominal organs |
| Inference latency | ~150–300ms per slice (GPU) |

## Performance (Latest Stable Run)

| Metric | Value |
|---|---|
| Mean Dice (Synapse, 9-class, excl. background) | 87.88% |
| Parameters | ~35–40M |
| FLOPs | ~28B |
| Training hardware | 1× NVIDIA Tesla T4 |
| Training time | ~10.7 hours (219 epochs, best checkpoint @ epoch 169) |

### Per-Organ Dice

| Organ | Dice (%) |
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

## Training Improvements (Research Notes)

Since the initial showcase release, iterative refinement of the training recipe (loss design, data augmentation strategy, and model regularization) closed a significant performance gap on historically hard classes:

- Gallbladder and Esophagus Dice improved from ~66–72% to ~88%, now on par with larger organs
- Overall mean Dice improved from an early ~80–82% baseline to a measured **87.88%**
- Exact augmentation recipe, loss weighting, and regularization settings are part of the private codebase

## Limitations

- Trained and evaluated only on Synapse Multi-Organ CT dataset
- Performance on MRI, ultrasound, or other modalities is untested
- Not validated for clinical use

## License

CC BY-NC 4.0 — Non-commercial research use only.
