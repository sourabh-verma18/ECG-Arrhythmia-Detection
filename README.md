# ECG-Arrhythmia-Detection
This README documents the Transformer-based workflow implemented in [Arthmyia_Detection_RP.ipynb](Arthmyia_Detection_RP.ipynb).

## What This Notebook Implements

This notebook trains an ECG arrhythmia classifier using:
- EWT (Empirical Wavelet Transform) decomposition into frequency modes
- A dual-branch patch embedding block:
  - Branch 1: 1D CNN on reconstructed raw signal
  - Branch 2: EWT mode channels
- Fusion of both branches, then Transformer encoder blocks for classification

Target classes:
- Class 0: Normal (non-VF)
- Class 1: VF/VT
- Class 2: Noisy

## Architecture Summary

Main model flow in [Arthmyia_Detection_RP.ipynb](Arthmyia_Detection_RP.ipynb):

1. Preprocess ECG windows (5 s at 250 Hz, 1250 samples).
2. Apply EWT to each window to obtain `NUM_MODES` channels.
3. Reconstruct raw-like signal from EWT (`sum(modes)`) and pass through a small 1D CNN branch.
4. Concatenate CNN features and EWT channels.
5. Patch projection with Conv1D (`kernel_size=patch_size`, `stride=patch_size`).
6. Add CLS token + positional embeddings.
7. Run stacked Transformer encoder blocks.
8. Use CLS token head for 3-class prediction.

Core settings used in the notebook:
- `TARGET_FS = 250`
- `WIN_SEC = 5`
- `NUM_MODES = 3`
- `PATCH_SIZE = 50`
- `D_MODEL = 256`
- `N_HEADS = 8`
- `N_LAYERS = 6`
- `D_FF = 1024`
- `EPOCHS = 50`

## Files and Outputs

Primary notebook:
- [Arthmyia_Detection_RP.ipynb](Arthmyia_Detection_RP.ipynb)

Expected generated outputs after running all training/evaluation cells:
- `best_vmd_transformer.pth` (best checkpoint)
- `training_curves.png`
- `confusion_matrix.png`
- `tsne_features.png`
- `attention_visualization.png`

Note: The checkpoint/output naming uses `vmd` in places, but this notebook uses EWT functions (`ewtpy.EWT1D`).

## Data Requirements

Notebook expects PhysioNet-style extracted folders containing record files (`.dat`, `.hea`, etc.) for:
- MIT-BIH Arrhythmia (normal class source)
- MIT-BIH Malignant Ventricular Ectopy (VF class source)
- CU Ventricular Tachyarrhythmia (VF class source)
- MIT-BIH Noise Stress Test (noise templates)

Inside the notebook, data paths are set near the top in the configuration cell.

## How To Run (Google Colab)

1. Open [Arthmyia_Detection_RP.ipynb](Arthmyia_Detection_RP.ipynb) in Colab.
2. Run the first cell to mount Google Drive.
3. Run the package install cell:
   - `pip install ewtpy wfdb`
4. Update data path variables in the configuration cell to your Drive folders.
5. Run all cells top to bottom.
6. Download output files/checkpoints from the runtime or Drive.

## How To Run (Local VS Code / Jupyter)

1. Create and activate a Python environment.
2. Install dependencies from [requirements.txt](requirements.txt) and also install `ewtpy` if not already present:
   - `pip install -r requirements.txt`
   - `pip install ewtpy`
3. Open [Arthmyia_Detection_RP.ipynb](Arthmyia_Detection_RP.ipynb).
4. Edit dataset path variables to local folders.
5. Remove or skip Colab-only cells (`google.colab` mount).
6. Run cells in order.

## Training Pipeline Details

1. Load records with `wfdb`.
2. Preprocess:
   - Resample to 250 Hz
   - High-pass filter for baseline drift
   - Z-score normalization
   - Segment into non-overlapping 5-second windows
3. Build synthetic noisy class from normal segments + NSTDB templates.
4. Apply EWT decomposition to all windows.
5. Train/validation split with stratification.
6. Train Transformer with:
   - AdamW optimizer
   - Warmup + cosine LR schedule
   - Gradient clipping
7. Evaluate with classification report and confusion matrix.
8. Generate interpretability plots:
   - t-SNE feature distribution
   - Attention overlays on ECG timeline

## Known Notes

- The notebook defines multiple model classes. The active training path uses the EWT Transformer model instantiated in the "BUILDING EWT-ONLY TRANSFORMER MODEL" section.
- Some plot titles/summary strings mention "VMD" while decomposition code uses EWT.
- Runtime and memory usage can be high on CPU; GPU is recommended.

## Troubleshooting

- `ModuleNotFoundError: ewtpy`:
  - Install with `pip install ewtpy`.
- `wfdb` record loading errors:
  - Confirm full record files exist together (`.dat`, `.hea`, etc.).
- CUDA unavailable:
  - Notebook will run on CPU but slower.
- Out-of-memory:
  - Reduce `BATCH_SIZE`, `MAX_SAMPLES`, or `D_MODEL`.

## Medical Disclaimer

This project is for research/educational use only and is not a clinical diagnostic system.
