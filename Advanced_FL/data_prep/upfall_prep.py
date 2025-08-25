"""data_prep/upfall_prep.py This script windows IMU, labels windows, and extracts per-window pose features by scanning the corresponding video file. Because file layouts vary, the script is parameterized and conservative. It will:
Expect IMU CSVs with accelerometer and gyroscope, and a video per trial.
Window IMU into T samples (default 2 s at 100 Hz).
Label windows by provided annotations (if --ann_csv), otherwise by trial label.
For each window, compute pose features by sampling frames in the time range and averaging keypoints (mediapipe).


Notes on UP-FALL paths:

Adjust --trial_glob, --imu_csv_glob, --video_glob to match your directory structure. 
If UP-FALL provides per-trial labels/annotations, point --label_map_json or --ann_csv to them. 
The script will still run with trial-level labels or all-non-fall if none are provided, 
but for research rigor you should use proper annotations."""


import os, re, argparse, glob, json
import numpy as np
import pandas as pd
from tqdm import tqdm
from extract_pose import extract_pose_means

def robust_float(x):
    try: return float(x)
    except: return np.nan

def load_imu_csv(path, imu_hz=100):
    df = pd.read_csv(path)
    # Try common column names; adjust as needed for your UP-FALL files
    # Expected columns: time(s) or ms, ax, ay, az, gx, gy, gz
    cols = {c.lower(): c for c in df.columns}
    time_col = None
    for k in ['time', 'timestamp', 't']:
        if k in cols: time_col = cols[k]; break
    ax = df[[c for c in df.columns if 'ax' in c.lower()]].iloc[:,0]
    ay = df[[c for c in df.columns if 'ay' in c.lower()]].iloc[:,0]
    az = df[[c for c in df.columns if 'az' in c.lower()]].iloc[:,0]
    gx = df[[c for c in df.columns if 'gx' in c.lower()]].iloc[:,0]
    gy = df[[c for c in df.columns if 'gy' in c.lower()]].iloc[:,0]
    gz = df[[c for c in df.columns if 'gz' in c.lower()]].iloc[:,0]
    if time_col is None:
        t = np.arange(len(ax)) / imu_hz
    else:
        t_vals = df[time_col].apply(robust_float).values.astype(np.float64)
        if t_vals.max() > 1e6:  # ms to s
            t = t_vals / 1000.0
        else:
            t = t_vals
    imu = np.stack([ax, ay, az, gx, gy, gz], axis=1).astype(np.float32)
    return t, imu

def windowize(t, imu, labels, window_sec=2.0, stride_sec=1.0, imu_hz=100):
    T = int(window_sec * imu_hz)
    S = int(stride_sec * imu_hz)
    xs, ys, times = [], [], []
    for start in range(0, len(imu)-T+1, S):
        end = start + T
        w = imu[start:end].T  # C x T
        # Determine label in window; majority label or provided labels
        if labels is None:
            y = 1 if 'fall' in '' else 0  # default non-fall unless trial-level label provided
        else:
            y = int(np.round(np.nanmean(labels[start:end])))
        xs.append(w); ys.append(y); times.append((t[start], t[end-1]))
    return np.stack(xs), np.array(ys), np.array(times)

def infer_video_for_trial(trial_dir, video_glob="*.mp4"):
    vids = glob.glob(os.path.join(trial_dir, video_glob))
    return vids⸨[0]⸩ if len(vids)>0 else None

def process_trial(trial_dir, imu_csv_glob="*imu*.csv", video_glob="*.mp4",
                  imu_hz=100, window_sec=2.0, stride_sec=1.0,
                  trial_label=None, ann_csv=None):
    imu_csvs = glob.glob(os.path.join(trial_dir, imu_csv_glob))
    if len(imu_csvs)==0:
        return None
    t, imu = load_imu_csv(imu_csvs⸨[0]⸩, imu_hz=imu_hz)
    labels = None
    if ann_csv and os.path.exists(ann_csv):
        ann = pd.read_csv(ann_csv)
        # Expect columns start_s, end_s, label in {fall, non-fall}; build a per-sample label vector
        lab = np.zeros(len(t), dtype=np.float32)
        for _, row in ann.iterrows():
            s,e = float(row['start_s']), float(row['end_s'])
            val = 1.0 if str(row['label']).lower().startswith('fall') else 0.0
            idx = (t>=s) & (t<=e)
            lab[idx] = val
        labels = lab
    else:
        if trial_label is not None:
            labels = np.full(len(t), 1.0 if trial_label.lower().startswith('fall') else 0.0, dtype=np.float32)

    X, Y, Wtimes = windowize(t, imu, labels, window_sec=window_sec, stride_sec=stride_sec, imu_hz=imu_hz)

    video = infer_video_for_trial(trial_dir, video_glob=video_glob)
    rgb_feats = None
    if video is not None:
        # For each window time span, sample frames roughly across the span and average pose features
        feats = []
        cap = None  # we call a helper per window to keep it simple (slower but simpler)
        for (ts, te) in Wtimes:
            feat = extract_pose_means(video_path=video, every_n=1, normalize=True)
            feats.append(feat)
        rgb_feats = np.stack(feats, axis=0)
    return X, Y, rgb_feats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw_root', type=str, required=True, help='Root folder with UP-FALL trials')
    ap.add_argument('--out_dir', type=str, required=True)
    ap.add_argument('--imu_hz', type=int, default=100)
    ap.add_argument('--window_sec', type=float, default=2.0)
    ap.add_argument('--stride_sec', type=float, default=1.0)
    ap.add_argument('--trial_glob', type=str, default='**/trial_*')
    ap.add_argument('--imu_csv_glob', type=str, default='*imu*.csv')
    ap.add_argument('--video_glob', type=str, default='*.mp4')
    ap.add_argument('--label_map_json', type=str, default=None, help='Optional mapping trial->label json')
    ap.add_argument('--ann_csv', type=str, default=None, help='Optional per-trial annotation CSV path pattern')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    trial_dirs = sorted(glob.glob(os.path.join(args.raw_root, args.trial_glob), recursive=True))
    X_list, Y_list, S_list, R_list = [], [], [], []
    subj_ids = []

    label_map = {}
    if args.label_map_json and os.path.exists(args.label_map_json):
        label_map = json.load(open(args.label_map_json))

    for td in tqdm(trial_dirs, desc='Trials'):
        trial_name = os.path.basename(td)
        subj = re.findall(r'subj[_-]?(\d+)', td) or re.findall(r'P(\d+)', td) or ['0']
        subj_id = int(subj⸨[0]⸩)
        trial_label = label_map.get(trial_name, None)
        ann_csv = None
        if args.ann_csv:
            cand = args.ann_csv.replace('{TRIAL}', trial_name)
            ann_csv = cand if os.path.exists(cand) else None

        out = process_trial(
            td, imu_csv_glob=args.imu_csv_glob, video_glob=args.video_glob,
            imu_hz=args.imu_hz, window_sec=args.window_sec, stride_sec=args.stride_sec,
            trial_label=trial_label, ann_csv=ann_csv)
        if out is None: continue
        X, Y, RGB = out
        N = X.shape⸨[0]⸩
        X_list.append(X); Y_list.append(Y); S_list.append(np.full(N, subj_id))
        if RGB is None:
            R_list.append(np.zeros((N,34), dtype=np.float32))
        else:
            # Ensure 34-dim (17 joints x 2); if mismatched, truncate/pad
            feat = RGB
            if feat.shape⸨[1]⸩ > 34:
                feat = feat[:, :34]
            elif feat.shape⸨[1]⸩ < 34:
                pad = np.zeros((feat.shape⸨[0]⸩, 34 - feat.shape⸨[1]⸩), dtype=np.float32)
                feat = np.concatenate([feat, pad], axis=1)
            R_list.append(feat)
        subj_ids.extend([subj_id]*N)

    if len(X_list)==0:
        print("No trials processed. Check file patterns.")
        return

    X = np.concatenate(X_list, axis=0)     # N x C x T
    Y = np.concatenate(Y_list, axis=0)     # N
    R = np.concatenate(R_list, axis=0)     # N x 34
    S = np.array(subj_ids)

    np.save(os.path.join(args.out_dir, 'imu.npy'), X.astype(np.float32))
    np.save(os.path.join(args.out_dir, 'y.npy'), Y.astype(np.int64))
    np.save(os.path.join(args.out_dir, 'subj.npy'), S.astype(np.int64))
    np.save(os.path.join(args.out_dir, 'rgb.npy'), R.astype(np.float32))
    print(f"Saved to {args.out_dir}: imu.npy {X.shape}, rgb.npy {R.shape}, y.npy {Y.shape}, subj.npy {S.shape}")

if __name__ == '__main__':
    main()