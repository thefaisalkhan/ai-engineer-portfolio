"""
Phase 2 — Week 7: TensorFlow & Keras
=======================================
Covers: Keras functional API, training loops, TF Serving prep, compare with PyTorch
Job relevance: 98% of ML job postings list TF/Keras — must know BOTH TF and PyTorch
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers
from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from typing import Optional


# ── Reproducibility ────────────────────────────────────────────────────────────
tf.random.set_seed(42)
np.random.seed(42)


@dataclass
class TrainingResult:
    model_name: str
    val_accuracy: Optional[float]
    val_loss: float
    n_params: int
    n_epochs_trained: int
    best_epoch: int


# ── Binary Classification with Keras Functional API ──────────────────────────
def build_classifier(
    input_dim: int,
    hidden_units: list[int] = [256, 128, 64],
    dropout_rate: float = 0.3,
    l2_lambda: float = 1e-4,
) -> keras.Model:
    """
    Functional API is preferred over Sequential for production models:
    - Supports multiple inputs/outputs
    - Shared layers
    - Residual connections
    """
    inputs = keras.Input(shape=(input_dim,), name="features")

    x = inputs
    for i, units in enumerate(hidden_units):
        x = layers.Dense(
            units,
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_lambda),
            name=f"dense_{i}",
        )(x)
        x = layers.BatchNormalization(name=f"bn_{i}")(x)
        x = layers.Dropout(dropout_rate, name=f"dropout_{i}")(x)

    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="binary_classifier")
    return model


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 64,
) -> tuple[keras.Model, TrainingResult]:
    """
    Full training loop with:
    - Early stopping (prevents overfitting)
    - Learning rate scheduling (ReduceLROnPlateau)
    - Model checkpointing (saves best weights)
    """
    model = build_classifier(input_dim=X_train.shape[1])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )

    training_callbacks = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=0,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=0,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=training_callbacks,
        verbose=0,
    )

    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    best_val_acc = float(max(history.history["val_accuracy"]))
    best_val_loss = float(min(history.history["val_loss"]))

    result = TrainingResult(
        model_name="BinaryClassifier-FunctionalAPI",
        val_accuracy=best_val_acc,
        val_loss=best_val_loss,
        n_params=model.count_params(),
        n_epochs_trained=len(history.history["loss"]),
        best_epoch=best_epoch,
    )

    return model, result


# ── Regression Model ──────────────────────────────────────────────────────────
def build_regression_model(input_dim: int) -> keras.Model:
    """Regression with residual (skip) connections — a common production pattern."""
    inputs = keras.Input(shape=(input_dim,))

    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)

    # Residual block
    residual = layers.Dense(64)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Add()([x, residual])  # skip connection
    x = layers.Activation("relu")(x)

    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(1, name="price_prediction")(x)

    return keras.Model(inputs, outputs, name="regression_with_residual")


# ── Custom Training Step ──────────────────────────────────────────────────────
class CustomTrainer(keras.Model):
    """
    Custom training loop: full control over gradients.
    Used when you need: gradient clipping, custom losses, multi-task learning.
    """

    def __init__(self, base_model: keras.Model):
        super().__init__()
        self.base_model = base_model
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.mae_tracker = keras.metrics.MeanAbsoluteError(name="mae")

    def train_step(self, data):
        X, y = data
        with tf.GradientTape() as tape:
            y_pred = self.base_model(X, training=True)
            loss = keras.losses.huber(y, y_pred, delta=1.0)
            loss = tf.reduce_mean(loss)

        # Gradient clipping (prevents exploding gradients)
        grads = tape.gradient(loss, self.base_model.trainable_variables)
        clipped_grads, _ = tf.clip_by_global_norm(grads, clip_norm=1.0)
        self.optimizer.apply_gradients(
            zip(clipped_grads, self.base_model.trainable_variables)
        )

        self.loss_tracker.update_state(loss)
        self.mae_tracker.update_state(y, y_pred)
        return {"loss": self.loss_tracker.result(), "mae": self.mae_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.mae_tracker]


# ── Export for TF Serving ─────────────────────────────────────────────────────
def export_for_serving(model: keras.Model, export_path: str = "/tmp/tf_serving_model"):
    """
    Save in SavedModel format for TF Serving deployment.
    This is how TensorFlow models go to production.
    """
    # Add preprocessing signature
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, model.input_shape[1]], dtype=tf.float32, name="features")
    ])
    def serving_fn(features):
        return {"predictions": model(features)}

    tf.saved_model.save(
        model,
        export_path,
        signatures={"serving_default": serving_fn},
    )
    return export_path


# ── PyTorch vs TensorFlow comparison table ────────────────────────────────────
FRAMEWORK_COMPARISON = {
    "PyTorch": {
        "paradigm": "Eager execution by default",
        "deployment": "TorchServe, ONNX, TorchScript",
        "dominance": "Research, academia, custom architectures",
        "training_loop": "Explicit: manual zero_grad, backward, step",
        "debugging": "Python debugger works natively",
        "key_strength": "Flexibility, research velocity",
    },
    "TensorFlow/Keras": {
        "paradigm": "Graph execution (tf.function) + eager",
        "deployment": "TF Serving, TF Lite, TF.js, SavedModel",
        "dominance": "Production, enterprise, mobile/edge",
        "training_loop": "model.fit() OR custom via GradientTape",
        "debugging": "tf.debugging, TensorBoard",
        "key_strength": "Production ecosystem, mobile deployment",
    },
}


if __name__ == "__main__":
    print("=== TensorFlow Version ===")
    print(f"TF: {tf.__version__}, Keras: {keras.__version__}")
    print(f"GPUs: {tf.config.list_physical_devices('GPU')}")

    # ── Binary Classification ─────────────────────────────────────────────────
    print("\n=== Binary Classification (Breast Cancer) ===")
    cancer = load_breast_cancer()
    X, y = cancer.data.astype(np.float32), cancer.target

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)

    model, result = train_classifier(X_train, y_train, X_val, y_val, epochs=80)
    print(f"Val Accuracy: {result.val_accuracy:.4f}")
    print(f"Val Loss:     {result.val_loss:.4f}")
    print(f"Parameters:   {result.n_params:,}")
    print(f"Best epoch:   {result.best_epoch}/{result.n_epochs_trained}")

    # Full evaluation
    _, acc, auc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Final AUC:  {auc:.4f}")

    # ── Regression ────────────────────────────────────────────────────────────
    print("\n=== Regression with Residual Connections (California Housing) ===")
    housing = fetch_california_housing()
    Xh = housing.data.astype(np.float32)
    yh = housing.target.astype(np.float32)

    Xh_train, Xh_val, yh_train, yh_val = train_test_split(
        Xh, yh, test_size=0.2, random_state=42
    )
    scaler_h = StandardScaler()
    Xh_train = scaler_h.fit_transform(Xh_train).astype(np.float32)
    Xh_val = scaler_h.transform(Xh_val).astype(np.float32)

    reg_model = build_regression_model(input_dim=Xh_train.shape[1])
    reg_model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    reg_model.fit(
        Xh_train, yh_train,
        validation_data=(Xh_val, yh_val),
        epochs=30, batch_size=128, verbose=0,
        callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
    )
    _, mae = reg_model.evaluate(Xh_val, yh_val, verbose=0)
    print(f"Regression MAE: {mae:.4f} (in $100k units)")
    print(f"Parameters: {reg_model.count_params():,}")

    # ── Framework Comparison ──────────────────────────────────────────────────
    print("\n=== PyTorch vs TensorFlow/Keras ===")
    for framework, props in FRAMEWORK_COMPARISON.items():
        print(f"\n{framework}:")
        for k, v in props.items():
            print(f"  {k}: {v}")
