import os
from collections import Counter
import numpy as np
from common import KNNClassifier

SOURCES = [
    ("ASL Alphabet (Kaggle)", "features.npz"),
    ("Everyday-phrase datasets", "extra_features.npz"),
    ("Custom recorded signs", "custom_features.npz"),
]
IMAGES_PER_CLASS = 300   # same value as in the extract scripts

Xs, ys, src = [], [], []

print("=== DATASET SUMMARY ===")
for i, (name, fname) in enumerate(SOURCES):
    if not os.path.exists(fname):
        print(f"{name}: file not found ({fname})")
        continue
    d = np.load(fname)
    if len(d["X"]) == 0:
        print(f"{name}: empty")
        continue
    Xs.append(d["X"])
    ys.append(d["y"])
    src.append(np.full(len(d["y"]), i))
    n_classes = len(set(d["y"]))
    line = f"{name}: {len(d['y'])} samples, {n_classes} classes"
    if i < 2:
        rate = len(d["y"]) / (n_classes * IMAGES_PER_CLASS) * 100
        line += f"  (hand detected in about {rate:.1f}% of images, approximate)"
    print(line)

if not Xs:
    raise SystemExit("No feature files found. Run the extract scripts first.")

X = np.concatenate(Xs)
y = np.concatenate(ys)
s = np.concatenate(src)
print(f"TOTAL: {len(y)} samples, {len(set(y))} classes")

# Same stratified 80/20 split as train.py
rng = np.random.default_rng(42)
test_idx = []
for c in np.unique(y):
    idx = np.where(y == c)[0]
    rng.shuffle(idx)
    test_idx.extend(idx[:max(1, int(len(idx) * 0.2))])
mask = np.zeros(len(y), dtype=bool)
mask[test_idx] = True

model = KNNClassifier(k=5).fit(X[~mask], y[~mask])
preds = model.predict(X[mask])
yt = y[mask]
st = s[mask]

print("\n=== OFFLINE RESULTS ===")
print(f"Test samples: {mask.sum()}")
print(f"OVERALL ACCURACY: {(preds == yt).mean() * 100:.2f}%")

print("\n=== BY GROUP (macro precision / recall) ===")
for i, (name, _) in enumerate(SOURCES):
    gm = st == i
    if not gm.any():
        continue
    precs, recs = [], []
    for c in np.unique(yt[gm]):
        tp = np.sum((preds == c) & (yt == c))
        pred_n = np.sum(preds == c)
        act = np.sum(yt == c)
        precs.append(tp / pred_n if pred_n else 0.0)
        recs.append(tp / act if act else 0.0)
    acc = (preds[gm] == yt[gm]).mean() * 100
    print(f"{name}: accuracy {acc:.2f}%, precision {np.mean(precs):.2f}, "
          f"recall {np.mean(recs):.2f}")

print("\n=== MOST CONFUSED PAIRS (true -> predicted, count) ===")
wrong = Counter((a, b) for a, b in zip(yt, preds) if a != b)
if not wrong:
    print("No mistakes on the test set.")
for (a, b), n in wrong.most_common(8):
    print(f"{a} -> {b}: {n}")

print("\n=== WEAKEST SIGNS (recall below 0.90) ===")
found = False
for c in np.unique(yt):
    act = np.sum(yt == c)
    rec = np.sum((preds == c) & (yt == c)) / act
    if rec < 0.90:
        print(f"{c}: recall {rec:.2f} ({act} test samples)")
        found = True
if not found:
    print("None.")