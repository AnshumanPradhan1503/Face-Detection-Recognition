
import pickle
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


# Project paths
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"

MODEL_PATH = MODEL_DIR / "final_model.h5"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.p"
CASCADE_PATH = MODEL_DIR / "haarcascade_frontalface_default.xml"


def load_recognition_components():
    """Load the trained CNN, class names, and face detector."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"Class names file not found: {CLASS_NAMES_PATH}"
        )

    if not CASCADE_PATH.exists():
        raise FileNotFoundError(
            f"Haar Cascade file not found: {CASCADE_PATH}"
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASS_NAMES_PATH, "rb") as file:
        class_names = pickle.load(file)

    face_detector = cv2.CascadeClassifier(str(CASCADE_PATH))

    if face_detector.empty():
        raise RuntimeError("Could not load the Haar Cascade classifier.")

    return model, class_names, face_detector


def main():
    print("=" * 55)
    print("REAL-TIME FACE RECOGNITION")
    print("=" * 55)

    model, class_names, face_detector = load_recognition_components()

    print("Model loaded successfully.")
    print("Recognized classes:", class_names)
    print("Starting webcam...")

    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Fallback if the first camera-opening method fails
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError(
            "Could not open webcam. Check camera permissions or connection."
        )

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read a frame from the webcam.")
                break

            # Convert frame to grayscale for face detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_detector.detectMultiScale(
                gray_frame,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(50, 50),
            )

            for (x, y, width, height) in faces:
                # Extract the detected face
                face = gray_frame[y:y + height, x:x + width]

                # Match the preprocessing used during training
                face = cv2.resize(face, (100, 100))
                face = cv2.equalizeHist(face)
                face = face.astype("float32") / 255.0

                # CNN input shape: (batch, height, width, channels)
                face = np.expand_dims(face, axis=-1)
                face = np.expand_dims(face, axis=0)

                predictions = model.predict(face, verbose=0)
                predicted_index = int(np.argmax(predictions[0]))
                predicted_name = class_names[predicted_index]

                # Draw the face rectangle and predicted name
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + width, y + height),
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    predicted_name,
                    (x, max(y - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("Face Detection and Recognition", frame)

            # Press Q to close the webcam window
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Webcam closed.")


if __name__ == "__main__":
    main()
