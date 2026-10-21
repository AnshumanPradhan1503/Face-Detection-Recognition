
import cv2
import pickle
import numpy as np
from pathlib import Path

# Project directories
PROJECT_DIR = Path(__file__).resolve().parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

# The two people to recognize
PERSON_NAMES = ["anshuman", "kirti"]

# Image settings matching the original project
IMAGE_SIZE = (100, 100)


def prepare_face_image(image_path):
    """Read an image and convert it to a standardized grayscale face image."""
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Skipping unreadable image: {image_path.name}")
        return None

    # Convert RGB/color images to grayscale.
    # Grayscale images are also handled correctly.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect the face in full photographs.
    cascade_path = (
        PROJECT_DIR / "haarcascade_frontalface_default.xml"
    )
    if not cascade_path.exists():
        cascade_path = (
            PROJECT_DIR / "models" / "haarcascade_frontalface_default.xml"
        )

    face_cascade = cv2.CascadeClassifier(str(cascade_path))

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(40, 40),
    )

    if len(faces) > 0:
        # Use the largest detected face.
        x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
        gray = gray[y:y + h, x:x + w]

    # Standardize image dimensions and contrast.
    gray = cv2.resize(gray, IMAGE_SIZE)
    gray = cv2.equalizeHist(gray)

    return gray


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    images = []
    labels = []

    for label, person_name in enumerate(PERSON_NAMES):
        person_dir = RAW_DIR / person_name

        if not person_dir.exists():
            print(f"WARNING: Folder not found: {person_dir}")
            continue

        image_paths = sorted(
            path for path in person_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

        accepted = 0

        for image_path in image_paths:
            face_image = prepare_face_image(image_path)

            if face_image is None:
                continue

            images.append(face_image)
            labels.append(label)
            accepted += 1

        print(f"{person_name}: {accepted}/{len(image_paths)} images prepared")

    if not images:
        raise RuntimeError(
            "No images were processed. Check your data/raw folders "
            "and Haar Cascade XML file."
        )

    # Keep images as grayscale arrays; normalize later during training.
    images = np.asarray(images, dtype=np.uint8)
    labels = np.asarray(labels, dtype=np.int64)

    with open(PROCESSED_DIR / "images.p", "wb") as file:
        pickle.dump(images, file)

    with open(PROCESSED_DIR / "labels.p", "wb") as file:
        pickle.dump(labels, file)

    with open(PROCESSED_DIR / "class_names.p", "wb") as file:
        pickle.dump(PERSON_NAMES, file)

    print("\nPreprocessing complete.")
    print("Processed image shape:", images.shape)
    print("Total images:", len(images))
    print("Saved to:", PROCESSED_DIR)


if __name__ == "__main__":
    main()
