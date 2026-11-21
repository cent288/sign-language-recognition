import os
import sys
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
from common import normalize_landmarks

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXTRA_DIR = os.path.join(SCRIPT_DIR, "extra_datasets")

IMAGES_PER_CLASS = 300
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Set to a list to use only some classes, e.g. {"HELLO", "THANK_YOU", "YES"}
# Leave as None to use every class folder found.
ONLY_CLASSES = None

# Folder names that are dataset splits, not signs
SPLIT_NAMES = {"train", "test", "val", "valid", "validation"}

if not os.path.isdir(EXTRA_DIR):
    print(f"Folder not found: {EXTRA_DIR}")
    print("Create it and put each dataset inside, one folder per sign.")
    sys.exit(1)

hands = mp.solutions.hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5,
)

# Collect image paths per class label
class_files = {}
for dirpath, dirnames, filenames in os.walk(EXTRA_DIR):
    images = [f for f in filenames if f.lower().endswith(IMAGE_EXTS)]
    if not images:
        continue
    label = os.path.basename(dirpath).strip().upper().replace(" ", "_")
    if label.lower() in SPLIT_NAMES:
        continue
    if ONLY_CLASSES is not None and label not in ONLY_CLASSES:
        continue
    class_files.setdefault(label, []).extend(
        os.path.join(dirpath, f) for f in sorted(images)
    )

if not class_files:
    print("No images found inside extra_datasets.")
    print("Expected layout: extra_datasets/<dataset>/<SIGN_NAME>/image.jpg")
    sys.exit(1)

print("Classes found:", sorted(class_files))

X, y = [], []
for label in sorted(class_files):
    files = class_files[label][:IMAGES_PER_CLASS]
    found = 0
    for path in tqdm(files, desc=label):
        img = cv2.imread(path)
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        if result.multi_hand_landmarks:
            X.append(normalize_landmarks(result.multi_hand_landmarks[0]))
            y.append(label)
            found += 1
    print(f"  {label}: hand detected in {found}/{len(files)} images")

if len(X) == 0:
    print("No hands were detected in any image.")
    sys.exit(1)

X = np.array(X)
y = np.array(y)
np.savez(os.path.join(SCRIPT_DIR, "extra_features.npz"), X=X, y=y)
print(f"Saved {len(X)} samples across {len(set(y))} classes")