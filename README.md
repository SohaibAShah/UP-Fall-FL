# UP-Fall Federated Learning (Wrist IMU)

A beginner-friendly starter to go from centralized training to FedAvg (and beyond) using the UP-Fall dataset (wrist IMU, binary fall vs no_fall). This project simulates one client per subject for federated learning.

## Contents

- `scripts/prepare_upfall_wrist.py`: Converts raw per-subject CSVs into windowed NPZ tensors.
- `src/trainers/train_central.py`: Centralized 1D CNN baseline.
- `src/fl/server.py`: Flower server (supports FedAvg or FedAdam).
- `src/fl/run_client.py`: Flower client runner (one subject per client).
- `src/models/cnn1d.py`: Small 1D CNN model.
- `src/datasets/wrist_npz.py`: Dataset loader with cross-subject splits.

## Quick Start

### 1. Create Environment

Using Conda, set up a Python 3.10 environment:

```bash
conda create -y -n upfall_fl python=3.10
conda activate upfall_fl
pip install --upgrade pip
```

### 2. Install PyTorch

Install PyTorch for your specific CUDA version (or CPU). For CUDA 12.7:

```bash
pip install --index-url https://download.pytorch.org/whl/cu127 torch torchvision torchaudio
```

### 3. Install FL Framework and Utilities

Install dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. Verify Setup

Check if CUDA is available:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### 5. Prepare UP-Fall Data

Place UP-Fall wrist-IMU raw CSVs in `data/upfall/raw/` as `subject_XX.csv` (one per subject). Each CSV must have:

- Wrist accelerometer columns (`x`, `y`, `z`)
- Wrist gyroscope columns (`x`, `y`, `z`)
- A `label` column: `1` for fall, `0` for no_fall

If your CSV column names differ, run:

```bash
python scripts/inspect_columns.py data/upfall/raw/subject_01.csv
```

Then, edit `WRIST_COL_PATTERNS` in `scripts/prepare_upfall_wrist.py` to match your column names.

### 6. Convert to Windowed NPZ

Convert raw CSVs to windowed NPZ tensors:

```bash
python scripts/prepare_upfall_wrist.py
```

This outputs `data/upfall/processed/subject_XX.npz`.

### 7. Run Centralized Baseline

Train a centralized 1D CNN baseline:

```bash
python -m src.trainers.train_central --epochs 20 --bs 64 --lr 1e-3
```

### 8. Run Federated Learning (FedAvg)

- **Terminal 1 (Server)**:

```bash
python -m src.fl.server --rounds 30 --strategy fedavg
```

- **Terminal 2..N (Clients)**:

Spawn one client per subject ID, starting at 0.

**Option A**: Launch all clients found in `data/upfall/processed/`:

```bash
bash scripts/run_clients.sh
```

**Option B**: Launch a single client manually:

```bash
python -m src.fl.run_client --cid 0 --local_epochs 1 --bs 64 --lr 1e-3
```

### 9. Try Other Strategies

- **FedAdam**:

```bash
python -m src.fl.server --rounds 30 --strategy fedadam
```

- **FedProx** (with proximal weight `mu`):

```bash
python -m src.fl.server --rounds 30 --strategy fedavg --mu 0.01
```

## Next Steps

- Add SCAFFOLD
- Add FedBN (exclude BatchNorm parameters from aggregation)
- Implement personalized heads
- Implement Ditto/pFedMe
- Add Differential Privacy using Opacus

## Notes

- This repo assumes one subject per CSV. If your raw data format differs, adapt `scripts/prepare_upfall_wrist.py` accordingly.
- Adjust column patterns and sampling rate in `scripts/prepare_upfall_wrist.py` based on your UP-Fall dataset description [2,24].
- For GPU support, ensure PyTorch is installed with the correct CUDA version (e.g., 12.7 as shown above). Verify GPU usage with `nvidia-smi`.