A research-grade framework to reproduce and extend multimodal, modality-aware FL for fall detection. It supports:
- Horizontal FL (sensors/subjects as clients) with residual fusion + IMU-driven gating, Pareto client selection, personalization (adapters + online ELM), differential privacy (Opacus), robust aggregation, and communication compression.
- Vertical/Hybrid FL simulator with feature-partitioned training (IMU vs RGB).

This repo is inspired by ideas in the provided literature set: multimodal residual fusion and client selection ⸨[2]⸩, DP in FL ⸨[6]⸩, online ELM personalization ⸨[8]⸩, and communication-efficient updates ⸨[10]⸩. It is a fresh implementation for your PhD project.

## Quickstart

1) Create env and install
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

Sanity check with synthetic data
python train_hfl.py --config config.yaml --synthetic 1
Prepare UP-FALL
Obtain the UP-FALL dataset (follow dataset terms).
Run preprocessing to create IMU windows and RGB pose features (rgb.npy):
python data_prep/upfall_prep.py --raw_root /path/to/UP-FALL --out_dir data/upfall_proc \
  --imu_hz 100 --window_sec 2.0 --stride_sec 1.0 --pose_backend mediapipe
Train on UP-FALL (horizontal FL)
python train_hfl.py --config config.yaml --data_dir data/upfall_proc
Try vertical/hybrid FL simulator
python train_vfl.py --config config.yaml --data_dir data/upfall_proc
Worked example (notebook-style)
Open examples/worked_example.py in VS Code or Jupyter (as cells), or convert to ipynb via:
pip install jupytext
jupytext --to notebook examples/worked_example.py
jupyter notebook examples/worked_example.ipynb
Notes and scope
Secure aggregation/MPC/HE are not included (hooks are provided). DP is supported via Opacus.
Pose extraction uses MediaPipe by default (CPU/GPU), configurable for other backends.
For cross-dataset generalization, add loaders (e.g., UNIMIB/MobiAct) similar to UP-FALL.
Repro tips
Always use subject-wise splits; avoid window leakage.
Report mean and worst-client (10th percentile) F1, ε/δ for DP, and bytes/round for comm compression.
For edge timing/energy, time I/O + preprocessing + inference; use real devices when possible.
License: MIT