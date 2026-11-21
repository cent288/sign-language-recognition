import numpy as np


def normalize_landmarks(hand_landmarks):
    """Turn 21 MediaPipe landmarks into a 63-value vector that
    doesn't depend on where the hand is or how big it appears."""
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    pts = pts - pts[0]              # make wrist the origin
    scale = np.max(np.abs(pts))
    if scale > 0:
        pts = pts / scale
    return pts.flatten()


class KNNClassifier:
    """Pure NumPy k-Nearest Neighbors (no scikit-learn needed)."""

    def __init__(self, k=5):
        self.k = k
        self.X = None
        self.classes = None
        self.y_idx = None

    def fit(self, X, y):
        self.X = np.asarray(X, dtype=np.float32)
        self.classes, self.y_idx = np.unique(np.asarray(y), return_inverse=True)
        return self

    def predict(self, X, batch=512):
        X = np.atleast_2d(np.asarray(X, dtype=np.float32))
        train_sq = (self.X ** 2).sum(axis=1)
        k = min(self.k, len(self.X))
        out = []
        for i in range(0, len(X), batch):
            xb = X[i:i + batch]
            # squared distances between each test row and every training row
            d = (xb ** 2).sum(axis=1)[:, None] - 2 * xb @ self.X.T + train_sq[None, :]
            d = np.maximum(d, 0)
            idx = np.argpartition(d, k - 1, axis=1)[:, :k]
            nd = np.take_along_axis(d, idx, axis=1)
            weights = 1.0 / (np.sqrt(nd) + 1e-6)   # closer neighbors count more
            labels = self.y_idx[idx]
            votes = np.zeros((len(xb), len(self.classes)))
            rows = np.arange(len(xb))
            for j in range(k):
                np.add.at(votes, (rows, labels[:, j]), weights[:, j])
            out.append(self.classes[votes.argmax(axis=1)])
        return np.concatenate(out)

    def save(self, path):
        np.savez(path, X=self.X, y=self.classes[self.y_idx], k=self.k)

    @staticmethod
    def load(path):
        data = np.load(path)
        model = KNNClassifier(k=int(data["k"]))
        model.fit(data["X"], data["y"])
        return model