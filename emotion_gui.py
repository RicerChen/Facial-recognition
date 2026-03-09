import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from mediapipe.python.solutions import face_mesh_connections as fmc

import tkinter as tk
from tkinter import filedialog, messagebox
import threading

# ================== 模型与预处理 ==================

class CNN_LSTM_Fusion(nn.Module):
    def __init__(self, num_classes=7, lstm_hidden=128):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),               # 24x24
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),               # 12x12
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # 128x1x1
            nn.Dropout(0.3)
        )
        self.cnn_fc = nn.Linear(128, 128)

        self.lstm = nn.LSTM(
            input_size=2,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.lm_fc = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, img, lm_seq):
        x_img = self.cnn(img).flatten(1)
        x_img = self.cnn_fc(x_img)

        out, _ = self.lstm(lm_seq)
        x_lm = out[:, -1, :]
        x_lm = self.lm_fc(x_lm)

        x = torch.cat([x_img, x_lm], dim=1)
        return self.classifier(x)


REGION_CONNECTIONS = {
    "lips": fmc.FACEMESH_LIPS,
    "left_eye": fmc.FACEMESH_LEFT_EYE,
    "right_eye": fmc.FACEMESH_RIGHT_EYE,
    "left_eyebrow": fmc.FACEMESH_LEFT_EYEBROW,
    "right_eyebrow": fmc.FACEMESH_RIGHT_EYEBROW,
    "nose": fmc.FACEMESH_NOSE,
    "face_oval": fmc.FACEMESH_FACE_OVAL,
}
USE_REGIONS = ["lips","left_eye","right_eye","left_eyebrow","right_eyebrow","nose","face_oval"]

def connections_to_indices(connections):
    idx = set()
    for a, b in connections:
        idx.add(a); idx.add(b)
    return idx

selected_idx = sorted(set().union(*(connections_to_indices(REGION_CONNECTIONS[r]) for r in USE_REGIONS)))
M = len(selected_idx)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5
)

def extract_selected_landmarks(gray48, upscale=4):
    h, w = gray48.shape[:2]
    img = cv2.resize(gray48, (w*upscale, h*upscale), interpolation=cv2.INTER_CUBIC)
    H, W = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    res = face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        return None

    lm = res.multi_face_landmarks[0].landmark
    pts = np.zeros((M, 2), dtype=np.float32)
    for k, idx in enumerate(selected_idx):
        x = lm[idx].x * W / upscale
        y = lm[idx].y * H / upscale
        pts[k] = [x, y]
    return pts

def normalize_landmarks_like_train(pts):
    pts = pts.astype(np.float32)
    pts = pts - pts.mean(axis=0, keepdims=True)
    norm = np.linalg.norm(pts) + 1e-6
    return pts / norm

def preprocess_img_like_train(face_bgr):
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
    gray = cv2.equalizeHist(gray)
    x = (gray.reshape(1, 48, 48) / 255.0).astype(np.float32)
    return gray, x

mp_face_detection = mp.solutions.face_detection
face_det = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

def get_largest_face_bbox(frame_bgr):
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = face_det.process(rgb)
    if not res.detections:
        return None

    best = None
    best_area = 0
    for det in res.detections:
        bb = det.location_data.relative_bounding_box
        x1 = int(bb.xmin * w); y1 = int(bb.ymin * h)
        bw = int(bb.width * w); bh = int(bb.height * h)
        x2 = x1 + bw; y2 = y1 + bh

        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w-1, x2); y2 = min(h-1, y2)

        area = max(0, x2-x1) * max(0, y2-y1)
        if area > best_area:
            best_area = area
            best = (x1,y1,x2,y2)
    return best


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS = "best_model.pth"
NUM_CLASSES = 7
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

model = CNN_LSTM_Fusion(num_classes=NUM_CLASSES, lstm_hidden=128).to(DEVICE)
model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))
model.eval()

# ================== 核心识别循环（摄像头/视频共用） ==================

def run_emotion_from_source(source):
    """
    source = 0 表示摄像头
    source = 'xxx.mp4' 表示视频文件路径
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        messagebox.showerror("错误", "无法打开视频源")
        return

    win_name = "Real-time Emotion (CNN+LSTM)"
    print("按 'q' 退出视频窗口")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        bbox = get_largest_face_bbox(frame)
        if bbox is not None:
            x1,y1,x2,y2 = bbox
            face_roi = frame[y1:y2, x1:x2].copy()
            gray48, img_in = preprocess_img_like_train(face_roi)
            pts = extract_selected_landmarks(gray48, upscale=4)

            if pts is not None:
                pts = normalize_landmarks_like_train(pts)

                with torch.no_grad():
                    img_t = torch.from_numpy(img_in).unsqueeze(0).to(DEVICE)
                    lm_t  = torch.from_numpy(pts).unsqueeze(0).to(DEVICE)

                    logits = model(img_t, lm_t)
                    prob = torch.softmax(logits, dim=1)[0].cpu().numpy()
                    pred = int(prob.argmax())
                    conf = float(prob[pred])

                label = f"{EMOTIONS[pred]} {conf:.2f}"
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            else:
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
                cv2.putText(frame, "FaceMesh failed", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

        cv2.imshow(win_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyWindow(win_name)

# ================== Tkinter 图形界面 ==================

def start_webcam():
    threading.Thread(target=run_emotion_from_source, args=(0,), daemon=True).start()

def start_video_file():
    path = filedialog.askopenfilename(
        title="选择视频文件",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
    )
    if not path:
        return
    threading.Thread(target=run_emotion_from_source, args=(path,), daemon=True).start()

def main():
    root = tk.Tk()
    root.title("表情识别系统")
    root.geometry("400x250")

    title_label = tk.Label(root, text="CNN + LSTM 表情识别", font=("Microsoft YaHei", 16))
    title_label.pack(pady=20)

    btn_webcam = tk.Button(root, text="摄像头实时识别", font=("Microsoft YaHei", 12),
                           width=20, command=start_webcam)
    btn_webcam.pack(pady=10)

    btn_video = tk.Button(root, text="选择视频文件识别", font=("Microsoft YaHei", 12),
                          width=20, command=start_video_file)
    btn_video.pack(pady=10)

    btn_quit = tk.Button(root, text="退出程序", font=("Microsoft YaHei", 12),
                         width=20, command=root.quit)
    btn_quit.pack(pady=10)

    info_label = tk.Label(root, text="在弹出的视频窗口中按 'q' 退出", font=("Microsoft YaHei", 10))
    info_label.pack(pady=5)

    root.mainloop()

if __name__ == "__main__":
    main()