import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
# ========== M4 Mac mediapipe 0.10.x 正确适配 ==========
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 关键连接常量（直接从 mp_face_mesh 取，不要用 frozenset 变量点属性）
REGION_CONNECTIONS = {
    "lips": mp_face_mesh.FACEMESH_LIPS,
    "left_eye": mp_face_mesh.FACEMESH_LEFT_EYE,
    "right_eye": mp_face_mesh.FACEMESH_RIGHT_EYE,
    "left_eyebrow": mp_face_mesh.FACEMESH_LEFT_EYEBROW,
    "right_eyebrow": mp_face_mesh.FACEMESH_RIGHT_EYEBROW,
    "nose": mp_face_mesh.FACEMESH_CONTOURS,       # 没有 FACEMESH_NOSE，用 CONTOURS 包含鼻子轮廓
    "face_oval": mp_face_mesh.FACEMESH_FACE_OVAL,
}

USE_REGIONS = ["lips", "left_eye", "right_eye", "left_eyebrow", "right_eyebrow", "nose", "face_oval"]

def connections_to_indices(connections):
    idx = set()
    for a, b in connections:
        idx.add(a)
        idx.add(b)
    return idx

selected_idx = sorted(set().union(*(connections_to_indices(REGION_CONNECTIONS[r]) for r in USE_REGIONS)))
M = len(selected_idx)

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
        x1 = int(bb.xmin * w)
        y1 = int(bb.ymin * h)
        bw = int(bb.width * w)
        bh = int(bb.height * h)
        x2 = x1 + bw
        y2 = y1 + bh
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w-1, x2)
        y2 = min(h-1, y2)
        area = max(0, x2-x1) * max(0, y2-y1)
        if area > best_area:
            best_area = area
            best = (x1, y1, x2, y2)
    return best

# ========== M4 适配：启用 MPS 加速 ==========
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"M4 设备已启用：{DEVICE}")

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget,
    QFileDialog, QMessageBox, QStatusBar
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QImage, QPixmap
import sys
import os

WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_model.pth")
NUM_CLASSES = 7
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

class CNN_LSTM_Fusion(nn.Module):
    def __init__(self, num_classes=7, lstm_hidden=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
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

# 加载模型
model = CNN_LSTM_Fusion(num_classes=NUM_CLASSES, lstm_hidden=128).to(DEVICE)
try:
    model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))
    model.eval()
except FileNotFoundError:
    raise FileNotFoundError(f"权重文件未找到！请确保 {WEIGHTS} 存在")

# ================== PyQt6 视频识别线程 ==================
class EmotionRecogThread(QThread):
    frame_signal = pyqtSignal(np.ndarray)
    error_signal = pyqtSignal(str)

    def __init__(self, source):
        super().__init__()
        self.source = source
        self.is_running = True

    def run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.error_signal.emit("无法打开视频源（摄像头/文件）")
            return
        while self.is_running:
            ok, frame = cap.read()
            if not ok:
                break
            bbox = get_largest_face_bbox(frame)
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                face_roi = frame[y1:y2, x1:x2].copy()
                gray48, img_in = preprocess_img_like_train(face_roi)
                pts = extract_selected_landmarks(gray48, upscale=4)
                if pts is not None:
                    pts = normalize_landmarks_like_train(pts)
                    with torch.no_grad():
                        img_t = torch.from_numpy(img_in).unsqueeze(0).to(DEVICE)
                        lm_t = torch.from_numpy(pts).unsqueeze(0).to(DEVICE)
                        logits = model(img_t, lm_t)
                        prob = torch.softmax(logits, dim=1)[0].cpu().numpy()
                        pred = int(prob.argmax())
                        conf = float(prob[pred])
                    label = f"{EMOTIONS[pred]} {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "FaceMesh failed", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            self.frame_signal.emit(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        self.is_running = False

    def stop(self):
        self.is_running = False
        self.wait()

# ================== PyQt6 主窗口 ==================
class EmotionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CNN + LSTM 表情识别")
        self.setGeometry(100, 100, 800, 600)
        self.recog_thread = None

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title_label = QLabel("CNN + LSTM 表情识别")
        title_label.setStyleSheet("font-size: 20px; font-weight: bold; margin-bottom: 20px;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        self.cam_btn = QPushButton("摄像头实时识别")
        self.cam_btn.setStyleSheet("font-size: 14px; padding: 10px 20px; margin: 5px;")
        self.cam_btn.clicked.connect(self.start_cam)
        layout.addWidget(self.cam_btn)

        self.video_btn = QPushButton("选择视频文件识别")
        self.video_btn.setStyleSheet("font-size: 14px; padding: 10px 20px; margin: 5px;")
        self.video_btn.clicked.connect(self.select_video)
        layout.addWidget(self.video_btn)

        self.stop_btn = QPushButton("停止识别")
        self.stop_btn.setStyleSheet("font-size: 14px; padding: 10px 20px; margin: 5px; background-color: #ff4444; color: white;")
        self.stop_btn.clicked.connect(self.stop_recog)
        self.stop_btn.setEnabled(False)
        layout.addWidget(self.stop_btn)

        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        layout.addWidget(self.video_label)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪 - M4 MPS 加速已启用", 3000)

    def start_cam(self):
        self.start_recog(0)

    def select_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        if path:
            self.start_recog(path)

    def start_recog(self, source):
        if self.recog_thread and self.recog_thread.isRunning():
            self.recog_thread.stop()
        self.recog_thread = EmotionRecogThread(source)
        self.recog_thread.frame_signal.connect(self.update_video_frame)
        self.recog_thread.error_signal.connect(self.show_error)
        self.recog_thread.start()
        self.cam_btn.setEnabled(False)
        self.video_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_bar.showMessage("识别中 - 按 'q' 或点击停止按钮退出")

    def stop_recog(self):
        if self.recog_thread and self.recog_thread.isRunning():
            self.recog_thread.stop()
        self.cam_btn.setEnabled(True)
        self.video_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.video_label.clear()
        self.status_bar.showMessage("已停止识别", 3000)

    def update_video_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image).scaled(
            self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.video_label.setPixmap(pixmap)

    def show_error(self, msg):
        QMessageBox.critical(self, "错误", msg)
        self.stop_recog()

    def closeEvent(self, event):
        if self.recog_thread and self.recog_thread.isRunning():
            self.recog_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = EmotionGUI()
    window.show()
    sys.exit(app.exec())
