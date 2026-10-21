
import cv2
import os
import time

# Folder where your face images will be saved
PERSON_NAME = "Anshuman"
SAVE_DIR = os.path.join("data", "raw", PERSON_NAME)

# Number of face images to collect
MAX_IMAGES = 100

# Load Haar Cascade face detector
CASCADE_PATH = "haarcascade_frontalface_default.xml"

if not os.path.exists(CASCADE_PATH):
    CASCADE_PATH = os.path.join(
        "models", "haarcascade_frontalface_default.xml"
    )

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

if face_cascade.empty():
    raise FileNotFoundError(
        "Haar Cascade XML file not found. "
        "Place haarcascade_frontalface_default.xml "
        "in the project folder or models folder."
    )

os.makedirs(SAVE_DIR, exist_ok=True)

# Open laptop webcam
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    camera.release()
    camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open webcam.")

count = 0
last_saved = 0.0

print("Look at the camera and move your face slightly.")
print("Press Q to stop early.")

while count < MAX_IMAGES:
    success, frame = camera.read()

    if not success:
        print("Could not read from webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(60, 60)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Save one image about every 0.3 seconds
        now = time.time()
        if now - last_saved >= 0.3 and count < MAX_IMAGES:
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (100, 100))
            face = cv2.equalizeHist(face)

            filename = os.path.join(
                SAVE_DIR, f"{PERSON_NAME}_{count + 1}.jpg"
            )
            if cv2.imwrite(filename, face):
                count += 1
                last_saved = now

    cv2.putText(
        frame,
        f"Person: {PERSON_NAME} | Images: {count}/{MAX_IMAGES}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )

    cv2.imshow("Collect Face Data - Press Q to Stop", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

print(f"Collection finished. Images saved: {count}")
print(f"Saved in: {os.path.abspath(SAVE_DIR)}")
