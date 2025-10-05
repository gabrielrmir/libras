from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode as RunningMode
import cv2
import math

import utils
from landmarker import Landmarker
from camera import Camera
from dataset import Dataset
from options import dataset_path

capture_dir = 'data/capture'
label = 'a'
request_frame = False

cam = Camera()
cap = cam.cap
w, h = cam.size

running_mode = RunningMode.LIVE_STREAM
# running_mode = RunningMode.IMAGE
landmarker = Landmarker(running_mode)

dataset = Dataset(dataset_path)

def handle_input():
    global label, request_frame

    key = cv2.waitKey(1)
    if key == ord('l'):
        label = input('label>')
    elif key == ord('c') and landmarker.has_result():
        hand = landmarker.result.hand_world_landmarks[0]
        dataset.save(label, hand)
        # p = Path(capture_dir, label)
        # p.mkdir(mode=0o755, parents=True, exist_ok=True)
        # filename = uuid.uuid4().hex + '.jpg'
        # cv2.imwrite(str(p / filename), cropped_im)
    elif key == ord('u'):
        request_frame = True
    elif key == 27: # esc
        quit(0)

def draw_landmarker(frame):
    landmarker.detect(frame)
    if not landmarker.has_result(): return False

    utils.draw_hands(frame, landmarker)
    boxes = landmarker.get_hands_boundaries()
    for box in boxes:
        w = box[2]-box[0]
        h = box[3]-box[1]
        d = math.sqrt(w*w+h*h)
        utils.draw_box(frame, box, d*50)

    return True

def draw():
    global request_frame

    ret, frame = cap.read()
    if not ret:
        print("Unable to receive frame. Exiting...")
        quit(0)

    cv2.flip(frame, 1, frame)
    utils.draw_text(frame, 'Label: ' + label, (10,40))

    if running_mode != RunningMode.IMAGE or request_frame:
        capture_frame = frame.copy()
        draw_landmarker(capture_frame)
        cv2.imshow('Capture', capture_frame)
        request_frame = False

    cv2.imshow('Libras', frame)

def quit(exit_code = 0):
    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()
    exit(exit_code)

if __name__ == '__main__':
    while True:
        handle_input()
        draw()
