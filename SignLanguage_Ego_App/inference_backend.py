import cv2
import time
import threading
import numpy as np
import torch
import torch.nn as nn

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =========================
# CONFIG
# =========================
ESP32_STREAM_URL = "http://192.168.4.1/stream"

LABEL_MAP_PATH = "label_map.npy"
X_LANDMARKS_PATH = "X_landmarks.npy"
MODEL_PATH = "gesture_end2end.pt"
LANDMARKER_TASK = "hand_landmarker.task"

NUM_HANDS = 2
FPS = 15
IDLE_SECONDS = 2.0
IDLE_FRAMES = int(FPS * IDLE_SECONDS)
MIN_SEQ_LEN = 10

DEVICE = torch.device("cpu")

# =========================
# GLOBAL STATE (Flask reads these)
# =========================
latest_prediction = "..."
latest_confidence = 0.0
prediction_lock = threading.Lock()

# =========================
# LOAD TRAINING METADATA
# =========================
label_map = np.load(LABEL_MAP_PATH, allow_pickle=True).item()
X_train = np.load(X_LANDMARKS_PATH)
T_MAX = X_train.shape[1]

print("[INFO] Loaded label_map and X_landmarks")
print("[INFO] T_MAX =", T_MAX)

# =========================
# MEDIAPIPE LANDMARKER
# =========================
def build_landmarker():
    base_options = python.BaseOptions(model_asset_path=LANDMARKER_TASK)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=NUM_HANDS,
        running_mode=vision.RunningMode.IMAGE
    )
    return vision.HandLandmarker.create_from_options(options)

landmarker = build_landmarker()

# =========================
# YOUR EXACT TRAINED MODEL
# =========================
class End2EndGestureModel(nn.Module):
    def __init__(self, num_classes, input_dim=252, emb_dim=64, hidden_dim=128, num_layers=2):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, emb_dim),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(
            emb_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, lengths):
        x = self.encoder(x)
        out, _ = self.lstm(x)

        B, T, H2 = out.shape
        mask = (torch.arange(T, device=out.device)
                .unsqueeze(0) < lengths.unsqueeze(1))
        mask = mask.unsqueeze(-1).float()

        out = out * mask
        sum_out = out.sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        mean_out = sum_out / denom

        return self.classifier(mean_out)

# =========================
# LOAD MODEL
# =========================
NUM_CLASSES = len(label_map)

model = End2EndGestureModel(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

print("[INFO] End2EndGestureModel loaded successfully")

# =========================
# LANDMARK UTILITIES
# =========================
'''
def extract_frame_landmarks(result):
    if not result.hand_landmarks:
        return np.zeros(126, dtype=np.float32)

    vec = []
    for hand in result.hand_landmarks:
        for lm in hand:
            vec.extend([lm.x, lm.y, lm.z])

    while len(vec) < 126:
        vec.extend([0.0, 0.0, 0.0])

    return np.array(vec[:126], dtype=np.float32)
'''

def extract_frame_landmarks(result):
    # Always store:
    # first 63 -> LEFT hand
    # next 63  -> RIGHT hand
    vec = np.zeros((126,), dtype=np.float32)

    if result.hand_landmarks is None:
        return vec

    hands = result.hand_landmarks
    handedness = result.handedness

    for h in range(len(hands)):
        # Mediapipe often flips left/right in selfie mode, 
        # but you must match strictly what 4_infer_end2end did.
        label = handedness[h][0].category_name  # "Left" or "Right"
        offset = 0 if label == "Left" else 63

        for i, lm in enumerate(hands[h]):
            base = offset + i * 3
            vec[base + 0] = lm.x
            vec[base + 1] = lm.y
            vec[base + 2] = lm.z

    return vec

def remove_zero_frames(seq):
    return seq[np.any(seq != 0, axis=1)]

'''
def normalize_landmarks(seq):
    mean = seq.mean(axis=0, keepdims=True)
    std = seq.std(axis=0, keepdims=True) + 1e-6
    return (seq - mean) / std
'''

def normalize_landmarks(seq):
    seq = seq.copy()

    # Normalize each hand (Left: 0-63, Right: 63-126) independently
    for hand_idx in range(2):
        start = hand_idx * 63
        # Extract specific hand (T, 21 points, 3 coords)
        hand = seq[:, start:start + 63].reshape(seq.shape[0], 21, 3)

        # 1. Center relative to Wrist (point 0)
        wrist = hand[:, 0:1, :]
        hand = hand - wrist

        # 2. Scale based on Middle Finger MCP (point 9) size
        scale = np.linalg.norm(hand[:, 9, :], axis=1) + 1e-6
        scale = scale.reshape(-1, 1, 1)
        hand = hand / scale

        # Put it back into the sequence
        seq[:, start:start + 63] = hand.reshape(seq.shape[0], 63)

    return seq

def add_velocity_features(seq):
    vel = np.diff(seq, axis=0, prepend=seq[:1])
    return np.concatenate([seq, vel], axis=1)

def pad_to_tmax(seq, tmax):
    true_len = len(seq)
    if true_len >= tmax:
        return seq[:tmax], tmax
    pad = np.zeros((tmax - true_len, seq.shape[1]), dtype=np.float32)
    return np.vstack([seq, pad]), true_len

# =========================
# CAMERA + INFERENCE THREAD
# =========================
class CameraInference:
    def __init__(self):
        # Start as None. The 'update' thread will handle the connection later.
        self.cap = None 
        
        self.frame = None
        self.seq_buffer = []
        self.idle_count = 0
        self.running = True

        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        global latest_prediction, latest_confidence
        frame_count = 0 

        while self.running:
            # --- AUTO-RECONNECT LOGIC START ---
            # If the camera disconnected or never started, try to reconnect
            if self.cap is None or not self.cap.isOpened():
                # print("[INFO] Searching for ESP32 Camera...") # Optional log
                try:
                    self.cap = cv2.VideoCapture(ESP32_STREAM_URL, cv2.CAP_FFMPEG)
                except Exception:
                    pass
                
                if not self.cap or not self.cap.isOpened():
                    time.sleep(1.0) # Wait 1 sec before retrying
                    continue
                else:
                    print("[INFO] ESP32 Camera Connected Successfully!")
            # --- AUTO-RECONNECT LOGIC END ---

            # Read frame
            ret, frame = self.cap.read()
            
            # If reading failed, it means we lost connection (e.g., you switched WiFi back)
            if not ret:
                print("[WARNING] Connection lost/Frame empty. Retrying...")
                self.cap.release() # Close the broken connection
                self.cap = None    # Reset to trigger reconnection logic above
                time.sleep(0.5)
                continue

            # --- NORMAL PROCESSING RESUMES HERE ---
            frame_count += 1
            #if frame_count % 30 == 0:
            #    print(f"[INFO] Processing frame {frame_count}...")

            self.frame = frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            result = landmarker.detect(mp_image)
            lm_vec = extract_frame_landmarks(result)

            if np.any(lm_vec != 0):
                self.seq_buffer.append(lm_vec)
                self.idle_count = 0
            else:
                self.idle_count += 1

            if self.idle_count >= IDLE_FRAMES and len(self.seq_buffer) >= MIN_SEQ_LEN:
                seq = np.stack(self.seq_buffer)
                self.seq_buffer.clear()
                self.idle_count = 0

                seq = remove_zero_frames(seq)
                seq = normalize_landmarks(seq)
                seq = add_velocity_features(seq)
                seq, true_len = pad_to_tmax(seq, T_MAX)

                x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                lengths = torch.tensor([true_len]).to(DEVICE)

                with torch.no_grad():
                    logits = model(x, lengths)
                    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

                pred_id = int(np.argmax(probs))
                pred_label = label_map[pred_id]
                confidence = float(probs[pred_id])

                with prediction_lock:
                    latest_prediction = pred_label
                    latest_confidence = confidence

                print(f"[PRED] {pred_label} ({confidence*100:.2f}%)")

    def get_frame(self):
        if self.frame is None:
            return None
        _, jpeg = cv2.imencode(".jpg", self.frame)
        return jpeg.tobytes()

camera = CameraInference()

# =========================
# FLASK HELPERS
# =========================
def generate_jpeg():
    while True:
        frame = camera.get_frame()
        if frame is None:
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

def get_latest_prediction():
    with prediction_lock:
        return latest_prediction, latest_confidence



# =========================
# COMPATIBILITY WRAPPER
# =========================
class GestureSystem:
    def __init__(self):
        self.camera = camera

    @property
    def latest_prediction(self):
        pred, _ = get_latest_prediction()
        return pred

    @property
    def latest_confidence(self):
        _, conf = get_latest_prediction()
        return conf

    def generate_jpeg(self):
        return generate_jpeg()
