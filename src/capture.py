import cv2
import utils
import os

from landmarker import Landmarker
from dataset import Dataset

CAMERA_INDEX = 0
if "CAMERA_INDEX" in os.environ:
    try:
        CAMERA_INDEX = int(os.environ["CAMERA_INDEX"])
    except:
        print("[ERROR] Invalid camera index")
        exit(1)

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("Cannot open camera")
    exit(1)

landmarker = Landmarker()
dataset = Dataset("./data/dataset.csv")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Unable to receive frame. Exiting...")
        break

    landmarker.detect_async(frame)

    key = cv2.waitKey(1)
    if key == 27: # esc
        break

    if hasattr(landmarker.result, "hand_landmarks") and \
        len(landmarker.result.hand_world_landmarks):

        w,h = 640,360
        hands = landmarker.result.hand_landmarks
        for hand in hands:
            utils.draw_hand(frame,hand,(w,h))

        hands = landmarker.result.hand_world_landmarks
        for hand in hands:
            utils.draw_hand_lines(frame,hand,(w,h),(w//8,h//8))

        if key >= 97 and key <= 122:
            dataset.save(chr(key),hands[0])


    cv2.imshow('Libras', frame)


dataset.close()
landmarker.close()
cap.release()
cv2.destroyAllWindows()
