"""
Neural Network: From Scratch → PyTorch
Implements forward pass, backprop, and training loop manually,
then replicates in PyTorch to show what the framework abstracts.
"""

import numpy as np


# ─── Activations ──────────────────────────────────────────────────────────────

def relu(z):      return np.maximum(0, z)
def relu_grad(z): return (z > 0).astype(float)
def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# ─── Neural Network from Scratch ──────────────────────────────────────────────

class NeuralNetScratch:
    """
    2-layer feedforward network with ReLU hidden layer.
    Demonstrates: forward pass, cross-entropy loss, backprop, SGD.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, lr: float = 0.01):
        self.lr = lr
        # Xavier initialization — prevents vanishing/exploding gradients
        scale1 = np.sqrt(2.0 / input_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)
        self.W1 = np.random.randn(input_dim, hidden_dim) * scale1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * scale2
        self.b2 = np.zeros(output_dim)

    def forward(self, X: np.ndarray) -> tuple:
        z1 = X @ self.W1 + self.b1
        a1 = relu(z1)
        z2 = a1 @ self.W2 + self.b2
        out = softmax(z2)
        return out, (X, z1, a1, z2)

    def loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        n = len(y_true)
        log_probs = -np.log(y_pred[np.arange(n), y_true] + 1e-9)
        return log_probs.mean()

    def backward(self, cache: tuple, y_pred: np.ndarray, y_true: np.ndarray):
        X, z1, a1, z2 = cache
        n = len(y_true)

        # Output layer gradient
        dz2 = y_pred.copy()
        dz2[np.arange(n), y_true] -= 1
        dz2 /= n

        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0)

        # Hidden layer gradient
        da1 = dz2 @ self.W2.T
        dz1 = da1 * relu_grad(z1)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0)

        # SGD update
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 500, batch_size: int = 64):
        n = len(X)
        for epoch in range(epochs):
            # Mini-batch SGD
            idx = np.random.permutation(n)
            epoch_loss = 0.0
            for i in range(0, n, batch_size):
                batch_idx = idx[i:i + batch_size]
                Xb, yb = X[batch_idx], y[batch_idx]
                y_pred, cache = self.forward(Xb)
                epoch_loss += self.loss(y_pred, yb)
                self.backward(cache, y_pred, yb)

            if epoch % 100 == 0:
                y_pred_all, _ = self.forward(X)
                acc = (y_pred_all.argmax(axis=1) == y).mean()
                avg_loss = epoch_loss / (n / batch_size)
                print(f"  Epoch {epoch:4d}: loss={avg_loss:.4f}, acc={acc:.4f}")

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred, _ = self.forward(X)
        return (y_pred.argmax(axis=1) == y).mean()


# ─── PyTorch Equivalent ───────────────────────────────────────────────────────

def pytorch_demo(X_train, y_train, X_test, y_test):
    """Same network in PyTorch. Shows what the framework abstracts."""
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("  (PyTorch not installed — skipping PyTorch demo)")
        return

    X_tr = torch.FloatTensor(X_train)
    y_tr = torch.LongTensor(y_train)
    X_te = torch.FloatTensor(X_test)
    y_te = torch.LongTensor(y_test)

    model = nn.Sequential(
        nn.Linear(X_train.shape[1], 64),
        nn.ReLU(),
        nn.Linear(64, len(np.unique(y_train))),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=64, shuffle=True)

    for epoch in range(200):
        for Xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        preds = model(X_te).argmax(dim=1)
        acc = (preds == y_te).float().mean().item()
    print(f"\n=== PyTorch Network ===")
    print(f"Test accuracy: {acc:.4f}")


# ─── Demo ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X, y = make_classification(n_samples=2000, n_features=20, n_classes=3,
                                n_informative=10, random_state=42)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("=== Neural Network from Scratch ===")
    net = NeuralNetScratch(input_dim=20, hidden_dim=64, output_dim=3, lr=0.05)
    net.train(X_train, y_train, epochs=500)
    print(f"\nTest accuracy: {net.accuracy(X_test, y_test):.4f}")

    pytorch_demo(X_train, y_train, X_test, y_test)
