
import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
)

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "models"

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)


def load_dataset():
    """Load processed images, labels, and class names."""
    with open(PROCESSED_DIR / "images.p", "rb") as file:
        images = pickle.load(file)

    with open(PROCESSED_DIR / "labels.p", "rb") as file:
        labels = pickle.load(file)

    with open(PROCESSED_DIR / "class_names.p", "rb") as file:
        class_names = pickle.load(file)

    # Normalize pixels to [0, 1] and add the grayscale channel.
    images = images.astype("float32") / 255.0
    images = images[..., np.newaxis]

    # One-hot encode the class labels.
    labels = tf.keras.utils.to_categorical(
        labels, num_classes=len(class_names)
    )

    return images, labels, class_names


def build_model(num_classes):
    """Build the original small CNN architecture."""
    model = Sequential([
        tf.keras.Input(shape=(100, 100, 1)),
        Conv2D(30, (5, 5), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(15, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(50, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def main():
    print("=" * 55)
    print("FACE RECOGNITION — CNN TRAINING")
    print("=" * 55)

    images, labels, class_names = load_dataset()

    print("Dataset shape:", images.shape)
    print("Number of classes:", len(class_names))
    print("Class names:", class_names)
    print("Training for 10 epochs...")

    model = build_model(num_classes=len(class_names))

    model.summary()

    model.fit(
        images,
        labels,
        epochs=10,
        validation_split=0.1,
        shuffle=True,
        verbose=1,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "final_model.h5"
    model.save(model_path)

    # Save the class names in the same order as the training labels.
    with open(MODEL_DIR / "class_names.p", "wb") as file:
        pickle.dump(class_names, file)

    print("\nTraining complete.")
    print("Model saved to:", model_path)
    print("Class names saved to:", MODEL_DIR / "class_names.p")


if __name__ == "__main__":
    main()
