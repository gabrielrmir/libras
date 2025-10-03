import cv2
import os

class Camera():
    def __init__(self):
        CAMERA_INDEX = 0
        if "CAMERA_INDEX" in os.environ:
            try:
                CAMERA_INDEX = int(os.environ["CAMERA_INDEX"])
            except:
                print("[ERROR] Invalid camera index")
                exit(1)

        self.cap = cv2.VideoCapture(CAMERA_INDEX)

        if not self.cap.isOpened():
            print("Cannot open camera")
            exit(1)

        ret, frame = self.cap.read()
        if not ret: exit(1)

        h, w, _ = frame.shape
        self.size = (w, h)
