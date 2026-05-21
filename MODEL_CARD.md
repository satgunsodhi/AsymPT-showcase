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

## Performance

| Metric | Value |
|---|---|
| Mean Dice (Synapse, 9-class) | 80–82% |
| Mean HD95 | 18–23 mm |
| Parameters | 35–40M |
| FLOPs | ~28B |
| Train Dice | ~96% |
| Val Dice | ~88% |

## Synapse Dataset — 9 Classes

| Index | Organ |
|---|---|
| 0 | Background |
| 1 | Spleen |
| 2 | Right Kidney |
| 3 | Left Kidney |
| 4 | Gallbladder |
| 5 | Esophagus |
| 6 | Liver |
| 7 | Stomach |
| 8 | Aorta |

## Training Improvements (Latest)

- **Focal Loss** added to `CombinedLoss` for hard-example mining on small organs (Gallbladder, Esophagus); based on Lin et al. (2017)
- **Label smoothing** parameter added to `CombinedLoss` to reduce overconfident predictions
- **EMA (Exponential Moving Average)** model wrapper (`utils/ema.py`) added for smoother weight averaging during training
- **Advanced augmentation pipeline** (`utils/augmentation.py`) targeting the train/val Dice gap (~96% vs ~88%):
  - Elastic deformation (soft tissue simulation, p=0.3)
  - Random scale + crop (multi-scale robustness 0.8–1.2×, p=0.5)
  - Cutout / random erasing (forces contextual learning, p=0.3)
  - Basic flips, rotations, brightness/contrast, Gaussian noise and blur

## Visualization

Updated `utils/visualization.py` includes:
- `visualize_predictions_grid`: curated grid selecting the most informative slices (highest organ count), with organ-colored overlays and per-slice Dice scores
- Error maps distinguishing: 🟢 correct foreground, 🔴 false negatives, 🟡 false positives, 🟠 misclassifications
- `compute_sample_dice`: per-class and mean Dice for single-sample evaluation

## Limitations

- Trained and evaluated only on Synapse Multi-Organ CT dataset
- Performance on MRI, ultrasound, or other modalities is untested
- Gallbladder and esophagus classes show lower Dice (~66–72%) due to small size and anatomical variability
- Not validated for clinical use

## License

CC BY-NC 4.0 — Non-commercial research use only.
