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

landmarker = Landmarker(RunningMode.LIVE_STREAM)
dataset = Dataset("./data/dataset.csv")

ret, frame = cap.read()
if not ret: exit(1)
h, w, _ = frame.shape


def draw_hands(frame):
    hands = landmarker.result.hand_landmarks
    for hand in hands:
        utils.draw_hand(frame,hand)

def handle_key(key):
    hand = landmarker.result.hand_world_landmarks[0]
    if key >= 97 and key <= 122:
        dataset.save(chr(key), hand)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Unable to receive frame. Exiting...")
        break

    landmarker.detect(frame)

    key = cv2.waitKey(1)
    if key == 27: # esc
        break

    if landmarker.has_result():
        handle_key(key)
        draw_hands(frame)
        boxes = landmarker.get_hands_boundaries()
        for box in boxes:
            utils.draw_box(frame, box)

    cv2.flip(frame, 1, frame)
    cv2.imshow('Libras', frame)

landmarker.close()
cap.release()
cv2.destroyAllWindows()
