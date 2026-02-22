"""
ML Algorithms from Scratch — Linear Regression, Logistic Regression, Decision Tree
No sklearn. Pure math. Builds intuition for what sklearn/PyTorch abstracts away.
"""

import numpy as np
from typing import Optional


# ─── Linear Regression ────────────────────────────────────────────────────────

class LinearRegression:
    """
    Gradient descent implementation.
    Key insight: every neural network layer is just a stacked linear regression.
    """

    def __init__(self, lr: float = 0.01, epochs: int = 1000):
        self.lr = lr
        self.epochs = epochs
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.loss_history: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for epoch in range(self.epochs):
            # Forward pass
            y_pred = X @ self.weights + self.bias

            # Loss: Mean Squared Error
            loss = np.mean((y_pred - y) ** 2)
            self.loss_history.append(loss)

            # Backward pass: gradients
            dw = (2 / n_samples) * (X.T @ (y_pred - y))
            db = (2 / n_samples) * np.sum(y_pred - y)

            # Parameter update
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights + self.bias

    def r2_score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1 - ss_res / ss_tot


# ─── Logistic Regression ──────────────────────────────────────────────────────

class LogisticRegression:
    """
    Binary classifier with sigmoid activation.
    Foundation: transformers output logits → softmax (multi-class sigmoid).
    """

    def __init__(self, lr: float = 0.1, epochs: int = 500):
        self.lr = lr
        self.epochs = epochs
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.epochs):
            z = X @ self.weights + self.bias
            y_pred = self._sigmoid(z)

            # Binary cross-entropy gradient
            dw = (1 / n_samples) * (X.T @ (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._sigmoid(X @ self.weights + self.bias)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return np.mean(self.predict(X) == y)


# ─── Decision Tree ────────────────────────────────────────────────────────────

class DecisionTree:
    """
    Recursive binary tree using information gain.
    Foundation: gradient boosting (XGBoost) stacks decision trees.
    """

    def __init__(self, max_depth: int = 5, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root: Optional[dict] = None

    def _gini(self, y: np.ndarray) -> float:
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple:
        best_gain, best_feat, best_thresh = -1, 0, 0.0
        parent_gini = self._gini(y)

        for feat in range(X.shape[1]):
            thresholds = np.unique(X[:, feat])
            for thresh in thresholds:
                left_mask = X[:, feat] <= thresh
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                n, n_l, n_r = len(y), left_mask.sum(), right_mask.sum()
                child_gini = (n_l / n) * self._gini(y[left_mask]) + \
                             (n_r / n) * self._gini(y[right_mask])
                gain = parent_gini - child_gini

                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, feat, thresh

        return best_feat, best_thresh

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> dict:
        # Leaf conditions
        if depth >= self.max_depth or len(y) < self.min_samples_split or len(np.unique(y)) == 1:
            return {"leaf": True, "value": int(np.bincount(y).argmax())}

        feat, thresh = self._best_split(X, y)
        left_mask = X[:, feat] <= thresh

        return {
            "leaf": False,
            "feature": feat,
            "threshold": thresh,
            "left": self._build(X[left_mask], y[left_mask], depth + 1),
            "right": self._build(X[~left_mask], y[~left_mask], depth + 1),
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTree":
        self.root = self._build(X, y.astype(int), depth=0)
        return self

    def _predict_one(self, x: np.ndarray, node: dict) -> int:
        if node["leaf"]:
            return node["value"]
        if x[node["feature"]] <= node["threshold"]:
            return self._predict_one(x, node["left"])
        return self._predict_one(x, node["right"])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_one(x, self.root) for x in X])

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return np.mean(self.predict(X) == y)


# ─── Demo ────────────────────────────────────────────────────────────────────

def demo():
    np.random.seed(42)
    n = 200

    # Linear regression: house price ≈ size × 300 + noise
    X_reg = np.random.rand(n, 2)
    y_reg = 300 * X_reg[:, 0] + 150 * X_reg[:, 1] + np.random.randn(n) * 20

    lr = LinearRegression(lr=0.1, epochs=1000).fit(X_reg, y_reg)
    print(f"=== Linear Regression ===")
    print(f"R² score: {lr.r2_score(X_reg, y_reg):.4f}")
    print(f"Final loss: {lr.loss_history[-1]:.4f}")

    # Logistic regression: binary classification
    X_clf = np.random.randn(n, 3)
    y_clf = (X_clf[:, 0] + X_clf[:, 1] > 0).astype(float)

    logr = LogisticRegression(lr=0.5, epochs=500).fit(X_clf, y_clf)
    print(f"\n=== Logistic Regression ===")
    print(f"Accuracy: {logr.accuracy(X_clf, y_clf):.4f}")

    # Decision tree
    X_tree = np.random.randn(n, 4)
    y_tree = (X_tree[:, 0] > 0).astype(int)

    dt = DecisionTree(max_depth=3).fit(X_tree, y_tree)
    print(f"\n=== Decision Tree ===")
    print(f"Accuracy: {dt.accuracy(X_tree, y_tree):.4f}")


if __name__ == "__main__":
    demo()
