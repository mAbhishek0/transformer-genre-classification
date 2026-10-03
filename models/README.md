# Model Weights

This directory contains trained model checkpoints (`.pt` files).
These files are **gitignored** due to their large size (250-570 MB each).

## Models

| File | Size | Architecture | Performance |
|------|------|-------------|-------------|
| `roberta_optimal_best.pt` | 476 MB | RoBERTa-base | **Best** — Test Macro-F1: 0.6870 |
| `distilbert_optimal_best.pt` | 253 MB | DistilBERT-base | Baseline — Test Macro-F1: 0.6831 |
| `longformer_best.pt` | 567 MB | Longformer-base-4096 | Long-context — Val Macro-F1: 0.6882 |
| `roberta_fusion_best.pt` | 476 MB | RoBERTa + Topic Fusion | Experiment — Test Macro-F1: 0.6828 |

## Extracting Model Weights

Due to GitHub's file size limits, these PyTorch model weights are stored as multi-part zip archives in the `models/splits/` directory.

To extract them:
1. Navigate to the `models/splits/` directory.
2. Use a tool like 7-Zip (Windows), The Unarchiver (Mac), or the command line to extract the `.zip` file for each model (e.g., `distilbert_optimal_best.zip`).
   - *Note: You only need to target the `.zip` file. The extractor will automatically stitch together the `.z01`, `.z02`, etc. parts.*
3. Move the extracted `.pt` files to this `models/` directory.
