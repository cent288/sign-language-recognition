import os
import sys
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
from common import normalize_landmarks

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Leave as None to auto-detect. If auto-detect fails, set it manually, e.g.
# DATA_DIR = r"C:\Users\rowel\Downloads\asl_alphabet_train\asl_alphabet_train"
DATA_DIR = None

IMAGES_PER_CLASS = 300
SKIP_CLASSES = {"nothing"}  # no hand in these images


def find_data_dir(root):
    """Find a folder that has class subfolders A, B and C inside it."""
    for dirpath, dirnames, _ in os.walk(root):
        # don't search inside the virtual environment
        dirnames[:] = [d for d in dirnames if d != ".venv"]
        if {"A", "B", "C"}.issubset(set(dirnames)):
            return dirpath
    return None


if DATA_DIR is None:
    DATA_DIR = find_data_dir(SCRIPT_DIR)

if DATA_DIR is None or not os.path.isdir(DATA_DIR):
    print("Could not find the dataset.")
    print(f"I searched inside: {SCRIPT_DIR}")
    print("Make sure the ASL Alphabet dataset is downloaded AND extracted")
    print("(not a .zip) into the sign_project folder, or set DATA_DIR")
    print("at the top of this file to the folder that contains A, B, C...")
    sys.exit(1)

print("Using dataset folder:", DATA_DIR)

hands = mp.solutions.hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5,
)

X, y = [], []

for label in sorted(os.listdir(DATA_DIR)):
    class_dir = os.path.join(DATA_DIR, label)
    if not os.path.isdir(class_dir) or label in SKIP_CLASSES:
        continue

    files = sorted(os.listdir(class_dir))[:IMAGES_PER_CLASS]
    for fname in tqdm(files, desc=label):
        img = cv2.imread(os.path.join(class_dir, fname))
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        if result.multi_hand_landmarks:
            X.append(normalize_landmarks(result.multi_hand_landmarks[0]))
            y.append(label)

if len(X) == 0:
    print("No hands were detected in any image. Check that the folder is correct.")
    sys.exit(1)

X = np.array(X)
y = np.array(y)
np.savez(os.path.join(SCRIPT_DIR, "features.npz"), X=X, y=y)
print(f"Saved {len(X)} samples across {len(set(y))} classes")