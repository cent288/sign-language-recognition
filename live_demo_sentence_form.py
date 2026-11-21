import os
import time
import textwrap
import subprocess
import cv2
import numpy as np
import mediapipe as mp
from collections import deque, Counter
from common import normalize_landmarks, KNNClassifier

SAMPLES_PER_SIGN = 200
COUNTDOWN_SECONDS = 3
HOLD_SECONDS = 1.0   # how long to hold a sign to add it to the sentence
FLIP = True          # must match FLIP in collect_custom.py

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


# ---------- sentence helpers ----------
def add_token(sentence, label):
    """Add a recognized sign to the sentence."""
    low = label.lower()
    if low == "space":
        if sentence and not sentence.endswith(" "):
            sentence += " "
        return sentence
    if low == "del":
        return sentence[:-1]
    if len(label) == 1:                      # a letter
        return sentence + label
    word = label.replace("_", " ")           # a whole-word sign
    if sentence and not sentence.endswith(" "):
        sentence += " "
    return sentence + word + " "


# ---------- drawing helpers ----------
def put(frame, text, y, color=(0, 255, 0), scale=0.8, thick=2):
    cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thick + 3)
    cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thick)


def draw_sentence_panel(frame, sentence):
    h, w = frame.shape[:2]
    panel_h = 120
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - panel_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    lines = textwrap.wrap(sentence, 32) or [""]
    lines = lines[-3:]
    lines[-1] += "_"                         # cursor
    y = h - panel_h + 38
    for line in lines:
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 255, 255), 2)
        y += 36


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

sentence = ""
candidate = None      # sign currently being held
candidate_since = 0.0
locked = False        # True once the held sign has been added
progress = 0.0


def flash(text, seconds=3):
    global message, message_until
    message = text
    message_until = time.time() + seconds


def speak(text):
    """Read the sentence aloud using Windows' built-in voice."""
    text = "".join(ch for ch in text if ch.isalnum() or ch == " ").strip()
    if not text:
        flash("Nothing to say yet")
        return
    cmd = [
        "powershell", "-NoProfile", "-Command",
        "Add-Type -AssemblyName System.Speech; "
        "(New-Object System.Speech.Synthesis.SpeechSynthesizer)"
        ".Speak('" + text + "')",
    ]
    try:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        subprocess.Popen(cmd, creationflags=flags)
        flash("Speaking...")
    except Exception:
        flash("Speech is not available on this computer")


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

            now = time.time()
            if text != candidate:
                candidate = text
                candidate_since = now
                locked = False
            held = now - candidate_since
            if not locked and held >= HOLD_SECONDS:
                sentence = add_token(sentence, text)
                locked = True
            progress = 1.0 if locked else min(held / HOLD_SECONDS, 1.0)
        else:
            history.clear()
            candidate = None
            locked = False