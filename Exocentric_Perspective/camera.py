import cv2
import numpy as np
import os
import mediapipe as mp
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import time

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        
        # --- MEDIAPIPE SETUP ---
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.7, 
            min_tracking_confidence=0.7
        )

        # --- LOAD DATA ---
        self.DATA_PATH = "Data_NPY"
        self.templates = self.load_reference_data()
        
        # --- VARIABLES ---
        self.sequence = []
        self.is_recording = False
        self.silence_start_time = None
        self.HAND_HEIGHT_THRESHOLD = 0.9  
        self.SILENCE_THRESHOLD = 1.0       
        self.MIN_FRAMES = 15     
        
        # This variable is read by app.py
        self.current_prediction = "" 

    def __del__(self):
        self.video.release()

    def load_reference_data(self):
        templates = []
        if not os.path.exists(self.DATA_PATH): 
            print("WARNING: Data_NPY folder not found.")
            return []
            
        # Load up to 20 actions (or more if you wish)
        target_actions = sorted(os.listdir(self.DATA_PATH))[:20]
        
        print(f"Loading {len(target_actions)} classes...")
        for action in target_actions:
            action_path = os.path.join(self.DATA_PATH, action)
            if not os.path.isdir(action_path): continue
            for f in os.listdir(action_path):
                if f.endswith('.npy'):
                    data = np.load(os.path.join(action_path, f))
                    templates.append((action, data))
        print(f"Loaded {len(templates)} total templates.")
        return templates

    def get_wrist_y(self, results):
        ys = []
        if results.pose_landmarks:
            ys.append(results.pose_landmarks.landmark[15].y)
            ys.append(results.pose_landmarks.landmark[16].y)
        if len(ys) > 0: return np.mean(ys)
        return 1.0

    # --- YOUR NEW FEATURE EXTRACTION LOGIC ---
    def get_hand_landmarks(self, results, hand_landmarks, wrist_idx):
        if hand_landmarks:
            return np.array([[res.x, res.y] for res in hand_landmarks.landmark]).flatten()
        
        if results.pose_landmarks:
            wrist = results.pose_landmarks.landmark[wrist_idx]
            # Fill 21 points with the wrist coordinate
            return np.tile([wrist.x, wrist.y], (21, 1)).flatten()
            
        return np.zeros(21*2)

    def extract_live_keypoints(self, results):
        if results.pose_landmarks:
            pose = np.array([[res.x, res.y] for res in results.pose_landmarks.landmark]).flatten()
        else:
            pose = np.zeros(33*2)
        
        # Left Hand (uses Left Wrist Index 15 if missing)
        lh = self.get_hand_landmarks(results, results.left_hand_landmarks, 15)
        # Right Hand (uses Right Wrist Index 16 if missing)
        rh = self.get_hand_landmarks(results, results.right_hand_landmarks, 16)
        
        return np.concatenate([pose, lh, rh])

    def normalize_live_sequence(self, sequence):
        seq_array = np.array(sequence)
        normalized = []
        
        for frame in seq_array:
            # Extract Pose landmarks (first 33*2 = 66 elements)
            pose_part = frame[:66]
            
            # Nose indices: 0, 1
            nose_x, nose_y = pose_part[0], pose_part[1]
            
            # Left Shoulder (22, 23) & Right Shoulder (24, 25)
            ls_x, ls_y = pose_part[22], pose_part[23]
            rs_x, rs_y = pose_part[24], pose_part[25]
            
            # Scale: Distance between shoulders
            shoulder_dist = np.linalg.norm(np.array([ls_x, ls_y]) - np.array([rs_x, rs_y]))
            scale = shoulder_dist if shoulder_dist > 0.05 else 1.0
            
            landmarks_xy = frame.reshape(-1, 2)
            
            # 1. Centering: Subtract Nose
            if nose_x != 0 and nose_y != 0:
                landmarks_xy -= [nose_x, nose_y]
                
            # 2. Scaling: Divide by shoulder width
            landmarks_xy /= scale
            
            normalized.append(landmarks_xy.flatten())
            
        return np.array(normalized)

    def get_frame(self):
        success, frame = self.video.read()
        if not success: return None, None

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw Landmarks
        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)
        self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
        self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)

        # --- LOGIC STATE MACHINE ---
        wrist_y = self.get_wrist_y(results)
        prediction_update = None 

        # 1. Recording Phase (Hands Up)
        if wrist_y < self.HAND_HEIGHT_THRESHOLD:
            if not self.is_recording:
                self.is_recording = True
                self.sequence = []
            self.silence_start_time = None
            
            # Use new extraction
            self.sequence.append(self.extract_live_keypoints(results))
            
            # UI Indicator (Green REC)
            cv2.circle(image, (30, 30), 15, (0, 255, 0), -1) 
            cv2.putText(image, "REC", (55, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 2. Processing Phase (Hands Down)
        else:
            if self.is_recording:
                if self.silence_start_time is None: self.silence_start_time = time.time()
                time_silence = time.time() - self.silence_start_time
                
                # UI Indicator (Orange Countdown)
                cv2.putText(image, "Processing...", (55, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

                if time_silence > self.SILENCE_THRESHOLD:
                    self.is_recording = False
                    if len(self.sequence) > self.MIN_FRAMES:
                        
                        # --- INFERENCE ---
                        try:
                            live_seq = self.normalize_live_sequence(self.sequence)
                            
                            best_score = float('inf')
                            best_action = None
                            
                            for action, template_data in self.templates:
                                # Optimization: Skip if length mismatch is huge
                                if abs(len(template_data) - len(live_seq)) > 30:
                                    continue

                                dist, _ = fastdtw(live_seq, template_data, radius=1, dist=euclidean)
                                if dist < best_score:
                                    best_score = dist
                                    best_action = action
                            
                            # Update the prediction for app.py
                            if best_action:
                                prediction_update = best_action
                                self.current_prediction = best_action
                                print(f"Detected: {best_action} ({best_score:.2f})")
                        
                        except Exception as e:
                            print(f"Inference Error: {e}")

                    self.sequence = []

        # Display Last Prediction on Video Feed
        if self.current_prediction:
            cv2.putText(image, f"Last: {self.current_prediction}", (10, 450), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes(), prediction_update
