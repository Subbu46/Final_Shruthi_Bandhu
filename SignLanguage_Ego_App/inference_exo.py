import os
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# ==============================
# CONFIG
# ==============================
DATA_DIR = "Data_NPY"

# DTW pruning threshold (same logic as your file)
MAX_LENGTH_DIFF_RATIO = 0.4

# ==============================
# LOAD REFERENCE DATA (ONCE)
# ==============================
def load_reference_data():
    reference_db = {}

    for action in os.listdir(DATA_DIR):
        action_path = os.path.join(DATA_DIR, action)
        if not os.path.isdir(action_path):
            continue

        samples = []
        for file in os.listdir(action_path):
            if file.endswith(".npy"):
                arr = np.load(os.path.join(action_path, file))
                samples.append(arr)

        if samples:
            reference_db[action] = samples

    print(f"[EXO] Loaded {len(reference_db)} gesture classes")
    return reference_db


REFERENCE_DB = load_reference_data()

# ==============================
# DTW MATCHING
# ==============================
def dtw_distance(seq1, seq2):
    dist, _ = fastdtw(seq1, seq2, dist=euclidean)
    return dist


# ==============================
# MAIN INFERENCE FUNCTION
# ==============================
def infer_exocentric(query_sequence: np.ndarray) -> str:
    """
    query_sequence: np.ndarray of shape (T, D)
    returns: best matching sentence (string)
    """

    best_action = None
    best_score = float("inf")

    query_len = len(query_sequence)

    for action, samples in REFERENCE_DB.items():
        for ref_seq in samples:
            ref_len = len(ref_seq)

            # ---- length pruning (same idea as your code)
            if abs(query_len - ref_len) / ref_len > MAX_LENGTH_DIFF_RATIO:
                continue

            score = dtw_distance(query_sequence, ref_seq)

            if score < best_score:
                best_score = score
                best_action = action

    if best_action is None:
        return "No confident match"

    return best_action
