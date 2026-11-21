import os
import numpy as np
from common import KNNClassifier

data = np.load("features.npz")
X, y = data["X"], data["y"]

# Optional extra feature files: dataset words and your own recordings
for fname in ["extra_features.npz", "custom_features.npz"]:
    if os.path.exists(fname):
        extra = np.load(fname)
        X = np.concatenate([X, extra["X"]])
        y = np.concatenate([y, extra["y"]])
        print(f"Added {fname}: {sorted(set(extra['y']))}")

print(f"Total samples: {len(X)}, classes: {len(set(y))}")

# Stratified 80/20 split (same idea as scikit-learn's train_test_split)
rng = np.random.default_rng(42)
test_idx = []
for c in np.unique(y):
    idx = np.where(y == c)[0]
    rng.shuffle(idx)
    n_test = max(1, int(len(idx) * 0.2))
    test_idx.extend(idx[:n_test])
test_mask = np.zeros(len(y), dtype=bool)
test_mask[test_idx] = True

X_train, y_train = X[~test_mask], y[~test_mask]
X_test, y_test = X[test_mask], y[test_mask]

# Evaluate on the held-out 20%
model = KNNClassifier(k=5).fit(X_train, y_train)
preds = model.predict(X_test)
print(f"\nAccuracy: {(preds == y_test).mean():.4f}\n")

print(f"{'class':<14}{'precision':>10}{'recall':>10}{'samples':>10}")
for c in np.unique(y):
    tp = np.sum((preds == c) & (y_test == c))
    predicted = np.sum(preds == c)
    actual = np.sum(y_test == c)
    precision = tp / predicted if predicted else 0.0
    recall = tp / actual if actual else 0.0
    print(f"{c:<14}{precision:>10.2f}{recall:>10.2f}{actual:>10}")

# Save the final model using ALL the data
final = KNNClassifier(k=5).fit(X, y)
final.save("sign_model.npz")
print("\nModel saved to sign_model.npz")