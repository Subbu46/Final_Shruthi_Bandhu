# infer.py  (TTS removed, camera index set to 1)
import os
import time
import json
import threading
from collections import deque, Counter
from threading import Lock

import cv2
import numpy as np

import torch
import torch.nn.functional as F
from torchvision import transforms, models
import torch.nn as nn

# ---------- Config ----------
MODEL_DIR = "C:/Users/chigu/stupidterminalgit/video_gesture_model_pytorch"
WEIGHT_FILE = "best_model.pth"
CAM_INDEX = 1             # changed to 1 as requested
NUM_FRAMES = 16
FRAME_SIZE = 224
CONF_THRESHOLD = 0.70
SMOOTHING_WINDOW = 5
DISPLAY_FPS_SMOOTH = 10
MIRROR = False
# ----------------------------

# --- Shared State for Threads ---
class SharedState:
    def __init__(self):
        self.frame_buffer = deque(maxlen=NUM_FRAMES)
        self.pred_buffer = deque(maxlen=SMOOTHING_WINDOW)
        self.conf_buffer = deque(maxlen=SMOOTHING_WINDOW)
        self.lock = Lock()
        self.running = True
        self.latest_prediction = "Waiting..."
        self.latest_confidence = 0.0

# ---------- Model definition ----------
class VideoGestureModel(nn.Module):
    def __init__(self, num_classes, num_frames=16):
        super(VideoGestureModel, self).__init__()
        self.num_frames = num_frames
        backbone = models.resnet50(pretrained=False)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        batch_size, num_frames, C, H, W = x.shape
        frame_features = []
        for i in range(num_frames):
            frame = x[:, i, :, :, :]
            feats = self.feature_extractor(frame)
            feats = feats.view(batch_size, -1)
            frame_features.append(feats)
        video_features = torch.stack(frame_features, dim=1)
        aggregated_features = torch.mean(video_features, dim=1)
        output = self.classifier(aggregated_features)
        return output
# -------------------------------------

# ---------- Helpers ----------
def load_mappings(mapping_path):
    """Load id2label mapping robustly. Accepts multiple key names and string/int keys."""
    with open(mapping_path, "r") as f:
        cfg = json.load(f)

    id2label = cfg.get("id2label") or cfg.get("idx_to_class") or cfg.get("idx_to_label") or cfg.get("id2class")
    label2id = cfg.get("label2id") or cfg.get("class_to_idx")

    if id2label is None and label2id is not None:
        try:
            id2label = {str(v): k for k, v in label2id.items()}
        except Exception:
            id2label = None

    if id2label is None:
        raise ValueError("No id2label or label2id found in class_mappings.json")

    id2label_int = {}
    for k, v in id2label.items():
        try:
            ik = int(k)
            id2label_int[ik] = v
        except Exception:
            try:
                id2label_int[int(str(k))] = v
            except:
                id2label_int[k] = v
    return id2label_int

def build_transform(frame_size):
    # Normalization transform (applied per-frame tensor)
    return transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

def preprocess_frames_fast(frame_deque, transform, device, num_frames, frame_size):
    frames = list(frame_deque)
    if len(frames) < num_frames:
        if not frames:
            black_frame = np.zeros((frame_size, frame_size, 3), dtype=np.uint8)
            frames = [black_frame] * num_frames
        else:
            frames = [frames[0]] * (num_frames - len(frames)) + frames

    processed_frames = []
    for frame in frames:
        resized_frame = cv2.resize(frame, (frame_size, frame_size), interpolation=cv2.INTER_AREA)
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb_frame).permute(2, 0, 1).float() / 255.0  # [C,H,W], values 0-1
        processed_frames.append(tensor)

    video_tensor = torch.stack(processed_frames, dim=0)
    video_tensor = torch.stack([transform(f) for f in video_tensor], dim=0)
    video_tensor = video_tensor.unsqueeze(0).to(device)  # [1, num_frames, C, H, W]
    return video_tensor

# ---------- Inference Thread ----------
class InferenceThread(threading.Thread):
    def __init__(self, model, transform, device, id2label, shared_state):
        super().__init__()
        self.model = model
        self.transform = transform
        self.device = device
        self.id2label = id2label
        self.shared_state = shared_state
        self.num_frames = NUM_FRAMES
        self.frame_size = FRAME_SIZE
        self.conf_threshold = CONF_THRESHOLD
        self.daemon = True

    def run(self):
        print("[Inference Thread] Starting inference loop...")
        while self.shared_state.running:
            with self.shared_state.lock:
                if len(self.shared_state.frame_buffer) < self.num_frames:
                    time.sleep(0.05)
                    continue
                frames_to_process = list(self.shared_state.frame_buffer)

            try:
                input_tensor = preprocess_frames_fast(
                    frames_to_process, self.transform, self.device, self.num_frames, self.frame_size
                )

                with torch.no_grad():
                    outputs = self.model(input_tensor)
                    probs = F.softmax(outputs, dim=1)
                    top_prob, top_idx = torch.max(probs, dim=1)
                    confidence = float(top_prob.item())
                    predicted_idx = int(top_idx.item())
                    predicted_label = self.id2label.get(predicted_idx, self.id2label.get(str(predicted_idx), "Unknown"))

            except Exception as e:
                print(f"[Inference Thread ERROR] Model processing failed: {e}")
                time.sleep(0.5)
                continue

            with self.shared_state.lock:
                self.shared_state.pred_buffer.append(predicted_label)
                self.shared_state.conf_buffer.append(confidence)

                if len(self.shared_state.pred_buffer) > 0:
                    most_common = Counter(self.shared_state.pred_buffer).most_common(1)[0][0]
                    avg_conf = sum(self.shared_state.conf_buffer) / len(self.shared_state.conf_buffer)
                else:
                    most_common = predicted_label
                    avg_conf = confidence

                self.shared_state.latest_confidence = avg_conf

                if avg_conf >= self.conf_threshold:
                    self.shared_state.latest_prediction = most_common
                else:
                    self.shared_state.latest_prediction = "Uncertain"

            time.sleep(0.01)

# ---------- Main ----------
def main():
    # Initialize shared state
    shared_state = SharedState()

    # Load mappings
    mapping_path = os.path.join(MODEL_DIR, "class_mappings.json")
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"{mapping_path} not found.")
    id2label = load_mappings(mapping_path)
    num_classes = len(id2label)

    # Load model weights (robust)
    weight_path = os.path.join(MODEL_DIR, WEIGHT_FILE)
    if not os.path.exists(weight_path):
        alt = os.path.join(MODEL_DIR, "final_model.pth")
        if os.path.exists(alt):
            weight_path = alt
        else:
            raise FileNotFoundError(f"Neither {WEIGHT_FILE} nor final_model.pth found in {MODEL_DIR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    model = VideoGestureModel(num_classes=num_classes, num_frames=NUM_FRAMES)
    checkpoint = torch.load(weight_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and any(k.startswith("module.") or k.startswith("backbone") or k.startswith("feature_extractor") for k in checkpoint.keys()):
        state_dict = checkpoint
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k
        if k.startswith("module."):
            new_key = k.replace("module.", "", 1)
        if new_key.startswith("backbone."):
            new_key = new_key.replace("backbone.", "", 1)
        new_state_dict[new_key] = v

    try:
        model.load_state_dict(new_state_dict, strict=False)
    except Exception as e:
        print(f"[WARN] load_state_dict(strict=False) raised: {e}. Attempting fallback.")
        try:
            model.load_state_dict(state_dict, strict=False)
        except Exception as e2:
            print(f"[ERROR] Failed to load weights: {e2}")
            raise

    model = model.to(device)
    model.eval()
    print(f"[INFO] Model loaded from {weight_path}")

    transform = build_transform(FRAME_SIZE)

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {CAM_INDEX}")

    inference_thread = InferenceThread(model, transform, device, id2label, shared_state)
    inference_thread.start()

    prev_time = time.time()
    fps = 0.0
    last_confident_prediction = ""

    print("[INFO] Starting webcam inference. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Camera read failed, retrying...")
                time.sleep(0.1)
                continue

            if MIRROR:
                frame = cv2.flip(frame, 1)

            with shared_state.lock:
                shared_state.frame_buffer.append(frame.copy())
                current_prediction = shared_state.latest_prediction
                current_confidence = shared_state.latest_confidence

            # Reset tracking when Uncertain
            if current_prediction == "Uncertain":
                last_confident_prediction = ""

            # Display text
            if current_prediction != "Waiting..." and current_prediction != "Uncertain":
                display_text = f"{current_prediction} ({current_confidence*100:.1f}%)"
                color = (0, 255, 0)
            else:
                display_text = current_prediction
                color = (0, 180, 255)

            # FPS smoothing
            cur_time = time.time()
            instantaneous_fps = 1.0 / (cur_time - prev_time + 1e-6)
            fps = (fps * (DISPLAY_FPS_SMOOTH - 1) + instantaneous_fps) / DISPLAY_FPS_SMOOTH
            prev_time = cur_time

            display_frame = cv2.resize(frame, (640, 480))
            cv2.putText(display_frame, f"Gesture: {display_text}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

            cv2.imshow("Gesture Inference - press q to quit", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")
    finally:
        with shared_state.lock:
            shared_state.running = False

        inference_thread.join(timeout=2.0)

        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Exited.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] An error occurred: {e}")
        cv2.destroyAllWindows()
