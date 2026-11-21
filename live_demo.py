import os
import time
import cv2
import numpy as np
import mediapipe as mp
from collections import deque, Counter
from common import normalize_landmarks, KNNClassifier

SAMPLES_PER_SIGN = 200
COUNTDOWN_SECONDS = 3
FLIP = True  # must match FLIP in collect_custom.py

CUSTOM_FILE = "custom_features.npz"
BASE_FILES = ["features.npz", "extra_features.npz"]
MODEL_FILE = "sign_model.npz"


# ---------- data helpers ----------
def load_npz(fname):
    if not os.path.exists(fname):
        return None
    d = np.load(fname)
    if len(d["X"]) == 0:
        return None
    return d["X"], d["y"]


def build_model():
    """Combine all feature files, fit the model, and save it."""
    Xs, ys = [], []
    for f in BASE_FILES + [CUSTOM_FILE]:
        data = load_npz(f)
        if data is not None:
            Xs.append(data[0])
            ys.append(data[1])
    if not Xs:
        return None
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    model = KNNClassifier(k=5).fit(X, y)
    model.save(MODEL_FILE)
    return model


def save_custom_samples(label, samples):
    new_X = np.array(samples, dtype=np.float32)
    new_y = np.array([label] * len(samples))
    old = load_npz(CUSTOM_FILE)
    if old is not None:
        new_X = np.concatenate([old[0], new_X])
        new_y = np.concatenate([old[1], new_y])
    np.savez(CUSTOM_FILE, X=new_X, y=new_y)


def delete_custom_label(label):
    """Remove one custom sign. Returns True if something was removed."""
    old = load_npz(CUSTOM_FILE)
    if old is None:
        return False
    X, y = old
    keep = y != label
    if keep.all():
        return False
    if keep.any():
        np.savez(CUSTOM_FILE, X=X[keep], y=y[keep])
    else:
        os.remove(CUSTOM_FILE)
    return True


def custom_labels():
    data = load_npz(CUSTOM_FILE)
    return sorted(set(data[1])) if data is not None else []


# ---------- drawing helper ----------
def put(frame, text, y, color=(0, 255, 0), scale=0.8, thick=2):
    cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thick + 3)
    cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thick)


# ---------- setup ----------
model = build_model()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

history = deque(maxlen=10)
cap = cv2.VideoCapture(0)

state = "live"        # live, typing, countdown, capture
action = None         # "add" or "delete" (used while typing)
typed = ""
label = ""
countdown_end = 0.0
samples = []
message = ""
message_until = 0.0


def flash(text, seconds=3):
    global message, message_until
    message = text
    message_until = time.time() + seconds


while True:
    ok, frame = cap.read()
    if not ok:
        break
    if FLIP:
        frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    hand = None
    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    # ---------- per-state logic and drawing ----------
    if state == "live":
        if hand is not None and model is not None:
            features = normalize_landmarks(hand).reshape(1, -1)
            history.append(model.predict(features)[0])
            text = Counter(history).most_common(1)[0][0]
        elif model is None:
            text = "Press N to add a sign"
        else:
            history.clear()
            text = "No hand"
        cv2.putText(frame, text, (30, 90), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (0, 255, 0), 4)
        put(frame, "N: add sign   D: delete custom sign   Q: quit",
            frame.shape[0] - 20, (255, 255, 255), 0.6, 1)

    elif state == "typing":
        verb = "NEW sign name" if action == "add" else "custom sign to DELETE"
        put(frame, f"Type {verb}:", 40, (0, 200, 255))
        put(frame, typed + "_", 85, (255, 255, 255), 1.0)
        put(frame, "Enter = confirm   Esc = cancel   Backspace = erase",
            frame.shape[0] - 20, (255, 255, 255), 0.6, 1)
        existing = custom_labels()
        if existing:
            put(frame, "Custom signs: " + ", ".join(existing),
                frame.shape[0] - 55, (200, 200, 200), 0.6, 1)

    elif state == "countdown":
        remaining = countdown_end - time.time()
        if remaining <= 0:
            samples = []
            history.clear()
            state = "capture"
        else:
            put(frame, f"Get ready to show: {label}", 40, (0, 200, 255))
            cv2.putText(frame, str(int(remaining) + 1), (30, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 200, 255), 6)

    elif state == "capture":
        if hand is not None:
            samples.append(normalize_landmarks(hand))
        put(frame, f"RECORDING {label}: {len(samples)}/{SAMPLES_PER_SIGN}",
            40, (0, 0, 255))
        put(frame, "Move your hand slightly closer/farther and tilt a bit",
            80, (255, 255, 255), 0.6, 1)
        if len(samples) >= SAMPLES_PER_SIGN:
            save_custom_samples(label, samples)
            model = build_model()
            history.clear()
            flash(f"Learned '{label}'!")
            state = "live"

    if time.time() < message_until:
        put(frame, message, 130 if state == "live" else frame.shape[0] - 90,
            (0, 255, 255))

    cv2.imshow("Sign Language Recognition", frame)

    # ---------- keyboard ----------
    key = cv2.waitKey(1) & 0xFF
    if key == 255:
        continue

    if state == "live":
        if key in (ord("q"), ord("Q")):
            break
        elif key in (ord("n"), ord("N")):
            state, action, typed = "typing", "add", ""
        elif key in (ord("d"), ord("D")):
            if custom_labels():
                state, action, typed = "typing", "delete", ""
            else:
                flash("No custom signs to delete")

    elif state == "typing":
        if key == 27:                       # Esc
            state = "live"
        elif key in (13, 10):               # Enter
            if not typed:
                state = "live"
            elif action == "add":
                label = typed
                countdown_end = time.time() + COUNTDOWN_SECONDS
                state = "countdown"
            else:
                if delete_custom_label(typed):
                    model = build_model()
                    flash(f"Deleted '{typed}'")
                else:
                    flash(f"'{typed}' not found in custom signs")
                state = "live"
        elif key in (8, 127):               # Backspace
            typed = typed[:-1]
        elif 32 <= key < 127 and len(typed) < 20:
            ch = chr(key).upper()
            if ch == " ":
                ch = "_"
            if ch.isalnum() or ch in "_-":
                typed += ch

    elif state in ("countdown", "capture"):
        if key == 27:                       # Esc cancels
            samples = []
            flash("Cancelled")
            state = "live"

cap.release()
cv2.destroyAllWindows()