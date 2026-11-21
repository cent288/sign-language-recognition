import sys
import os
import cv2
import numpy as np
import mediapipe as mp
from common import normalize_landmarks

# Usage: python collect_custom.py HELLO
if len(sys.argv) < 2:
    print("Usage: python collect_custom.py LABEL_NAME")
    sys.exit(1)

LABEL = sys.argv[1].upper()
SAMPLES_TO_COLLECT = 200
FLIP = True  # must match FLIP in live_demo.py
OUT_FILE = "custom_features.npz"

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

X_new, y_new = [], []
capturing = False
cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    if FLIP:
        frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
        if capturing:
            X_new.append(normalize_landmarks(hand))
            y_new.append(LABEL)

    if capturing and len(X_new) >= SAMPLES_TO_COLLECT:
        break

    status = "CAPTURING" if capturing else "Press C to start"
    cv2.putText(frame, f"{LABEL}: {len(X_new)}/{SAMPLES_TO_COLLECT}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, status, (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)
    cv2.imshow("Collect", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("c"):
        capturing = True
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

if len(X_new) == 0:
    print("No samples collected.")
    sys.exit(1)

X_new = np.array(X_new)
y_new = np.array(y_new)

# Append to existing custom data if the file already exists
if os.path.exists(OUT_FILE):
    old = np.load(OUT_FILE)
    X_new = np.concatenate([old["X"], X_new])
    y_new = np.concatenate([old["y"], y_new])

np.savez(OUT_FILE, X=X_new, y=y_new)
print(f"Saved. custom_features.npz now has {len(X_new)} samples, "
      f"classes: {sorted(set(y_new))}")