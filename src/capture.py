import cv2
import utils
import os

from landmarker import Landmarker
from dataset import Dataset
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode as RunningMode

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

landmarker = Landmarker(RunningMode.IMAGE)
dataset = Dataset("./data/dataset.csv")

w,h = 640,360
ret, frame = cap.read()
if not ret:
    print("Unable to receive frame. Exiting...")
    exit(1)

def check():
    if not landmarker.has_result(): return

    hands = landmarker.result.hand_landmarks
    for hand in hands:
        utils.draw_hand(frame,hand,(w,h))

    hands = landmarker.result.hand_world_landmarks
    for hand in hands:
        utils.draw_hand_lines(frame,hand,(w,h),(w//8,h//8))

while True:
    key = cv2.waitKey(1)
    ret, frame = cap.read()
    if key == 27: # esc
        break

    if key >= 97 and key <= 122:
        # dataset.save(chr(key),hands[0])
        if not ret:
            print("Unable to receive frame. Exiting...")
            break
        landmarker.detect(frame)
        check()
    cv2.imshow('Libras', frame)


dataset.close()
landmarker.close()
cap.release()
cv2.destroyAllWindows()
