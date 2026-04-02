# Topology Optimization — UNet

Topology optimization using a UNet model (training, inference, and a small Streamlit UI).

## Quick Start

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Train (example):

```powershell
python -m src.train.train --data data/dataset --epochs 20 --batch-size 32 --lr 1e-3
```

4. Run batch inference on the test set and save predictions:

```powershell
python -m src.inference.predict --checkpoint checkpoints/unet_best.pth --test-dir data/dataset/test --save-dir outputs/preds --viz-dir outputs/viz
```

5. Run the Streamlit UI for single-input inference:

```powershell
streamlit run scripts/streamlit_app.py
```

6. Generate a small example input `.npz` for testing:

```powershell
python scripts/generate_example_npz.py --out examples/sample_test.npz
```

## Input `.npz` format (for UI / `predict.py`)
- Required keys: `force_x_elem`, `force_y_elem`, `bc_elem` (all 2D arrays of same shape, numeric, preferably float32).
- Optional: `volfrac` (scalar) and `rho` (ground-truth mask). The Streamlit UI ignores `rho` for visualization unless you enable metric display.

## Notes
- Checkpoints (`checkpoints/*.pth`) are intentionally ignored in the Git history. Use Git LFS if you need to store large model files in the repo.
- For production deployment you may want to export the best model to TorchScript or ONNX (not included by default).

## Contact
Open an issue or edit files in the repository for changes.
