
import cv2


def main():
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not open the webcam.")
        print("Check camera permissions and close other apps using the camera.")
        return

    print("Webcam connected successfully!")
    print("Press Q to close the camera window.")

    while True:
        success, frame = camera.read()

        if not success:
            print("ERROR: Could not read a frame.")
            break

        cv2.imshow("Webcam Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
