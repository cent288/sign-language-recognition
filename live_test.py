import os

# Must be set BEFORE importing cv2: avoids a common Windows camera failure
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import sys
import csv
import time
import string
from collections import Counter
import cv2
import numpy as np
import mediapipe as mp
from common import normalize_landmarks, KNNClassifier

SKIP_MOTION_LETTERS = True   # J and Z need movement, so they are left out
FLIP = True                  # must match FLIP in live_demo.py
READY_SECONDS = 2.0          # time to get your hand in position
CAPTURE_SECONDS = 1.5        # window used to decide the prediction
MIN_FRAMES = 8               # need at least this many frames with a hand
RESULT_SECONDS = 1.5
WINDOW = "Live Test"

LETTERS = [c for c in string.ascii_uppercase
           if not (SKIP_MOTION_LETTERS and c in "JZ")]


# ---------- camera ----------
def open_camera():
    backends = [
        ("DirectShow", cv2.CAP_DSHOW),
        ("Media Foundation", cv2.CAP_MSMF),
        ("default", cv2.CAP_ANY),
    ]
    for index in (0, 1, 2):
        for name, backend in backends:
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                for _ in range(15):
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        print(f"Camera ready: index {index}, backend {name}")
                        return cap
                    time.sleep(0.1)
            cap.release()
    return None


# ---------- model ----------
def load_model():
    if os.path.exists("sign_model.npz"):
        return KNNClassifier.load("sign_model.npz")
    Xs, ys = [], []
    for f in ["features.npz", "extra_features.npz", "custom_features.npz"]:
        if os.path.exists(f):
            d = np.load(f)
            if len(d["X"]):
                Xs.append(d["X"])
                ys.append(d["y"])
    if not Xs:
        return None
    return KNNClassifier(k=5).fit(np.concatenate(Xs), np.concatenate(ys))


def put(frame, text, org, color=(0, 255, 0), scale=1.0, thick=2):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thick + 3)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thick)


# ---------- setup ----------
model = load_model()
if model is None:
    print("No model or feature files found. Run extract_features.py first.")
    sys.exit(1)

cap = open_camera()
if cap is None:
    print("Could not get frames from any camera. Close other apps using it.")
    sys.exit(1)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

results = []          # (letter, predicted, correct) or (letter, "skipped", None)
idx = 0
state = "wait"        # wait, ready, capture, result
preds = []
ready_end = cap_end = result_end = 0.0
message = ""
message_color = (255, 255, 255)
retry = False
failed_reads = 0
first_frame = True

while idx < len(LETTERS):
    ok, frame = cap.read()
    if not ok or frame is None:
        failed_reads += 1
        if failed_reads > 30:
            print("Lost the camera feed.")
            break
        time.sleep(0.03)
        continue
    failed_reads = 0

    if FLIP:
        frame = cv2.flip(frame, 1)

    if first_frame:
        fh, fw = frame.shape[:2]
        cv2.resizeWindow(WINDOW, int(fw * 1.3), int(fh * 1.3))
        first_frame = False

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    live_pred = None
    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
        features = normalize_landmarks(hand).reshape(1, -1)
        live_pred = model.predict(features)[0]

    letter = LETTERS[idx]
    now = time.time()

    # ---------- state logic ----------
    if state == "ready":
        if now >= ready_end:
            preds = []
            cap_end = now + CAPTURE_SECONDS
            state = "capture"

    elif state == "capture":
        if live_pred is not None:
            preds.append(live_pred)
        if now >= cap_end:
            if len(preds) < MIN_FRAMES:
                message = "No hand detected - try this letter again"
                message_color = (0, 165, 255)
                retry = True
            else:
                predicted = Counter(preds).most_common(1)[0][0]
                correct = predicted == letter
                results.append((letter, predicted, correct))
                message = (f"Predicted {predicted}: CORRECT" if correct
                           else f"Predicted {predicted}: WRONG")
                message_color = (0, 200, 0) if correct else (0, 0, 255)
                retry = False
            result_end = now + RESULT_SECONDS
            state = "result"

    elif state == "result":
        if now >= result_end:
            if not retry:
                idx += 1
            state = "wait"
            continue

    # ---------- drawing ----------
    h, w = frame.shape[:2]
    put(frame, f"{idx +