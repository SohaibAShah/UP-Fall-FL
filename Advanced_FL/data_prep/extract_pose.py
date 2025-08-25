import cv2, numpy as np
import mediapipe as mp

COCO17_FROM_MP = ⸨[0,2,5,7,8,11,12,13,14,15,16,23,24,25,26,27,28]⸩  # pick 17 reasonably stable indices

def extract_pose_means(video_path, every_n=1, normalize=True):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False)
    cap = cv2.VideoCapture(video_path)
    xs, ys = [], []
    while True:
        ret, frame = cap.read()
        if not ret: break
        if every_n>1 and int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % every_n != 0:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if res.pose_landmarks is None:
            continue
        lm = res.pose_landmarks.landmark
        pts = np.array([[lm[i].x, lm[i].y] for i in COCO17_FROM_MP], dtype=np.float32)  # 17x2
        if normalize:
            # normalize by bounding box of landmarks
            minv = pts.min(axis=0); maxv = pts.max(axis=0); scale = (maxv - minv + 1e-6)
            pts = (pts - minv) / scale
        xs.append(pts[:,0]); ys.append(pts[:,1])
    cap.release(); pose.close()
    if len(xs)==0:
        return np.zeros((17*2,), dtype=np.float32)
    xmean = np.mean(np.stack(xs), axis=0)
    ymean = np.mean(np.stack(ys), axis=0)
    feat = np.concatenate([xmean, ymean], axis=0)  # 34-dim
    return feat.astype(np.float32)