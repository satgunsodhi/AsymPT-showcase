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

## Limitations

- Trained and evaluated only on Synapse Multi-Organ CT dataset
- Performance on MRI, ultrasound, or other modalities is untested
- Gallbladder and esophagus classes show lower Dice (~66–72%) due to small size and anatomical variability
- Not validated for clinical use

## License

CC BY-NC 4.0 — Non-commercial research use only.
