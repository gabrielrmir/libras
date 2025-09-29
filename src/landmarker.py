from functools import reduce
from mediapipe.tasks.python.components.containers.landmark import Landmark
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerResult
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode as RunningMode
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerOptions
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker
import mediapipe as mp
import time
import utils

def _get_bounding_box(hand):
    hand = utils.hand_to_points_array(hand)
    if hand == None: return None
    pos = hand.pop()
    box = [pos[0], pos[1], pos[0], pos[1]]
    for pos in hand:
        box[0] = min(box[0], pos[0])
        box[1] = min(box[1], pos[1])
        box[2] = max(box[2], pos[0])
        box[3] = max(box[3], pos[1])
    return box

class Landmarker():
    def __init__(self, mode: RunningMode):
        self.result = HandLandmarkerResult
        self.landmarker = self.create_landmarker(mode)
        self.running_mode = mode
        self.detect = self._detect_async if mode == RunningMode.LIVE_STREAM else self._detect

    def create_landmarker(self, mode: RunningMode):
        def callback(result, output_image, timestamp_ms):
            self.result = result

        options = HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path='./models/hand_landmarker.task'),
            running_mode=mode,
            result_callback=callback if mode == RunningMode.LIVE_STREAM else None)

        return HandLandmarker.create_from_options(options)

    def _detect(self, frame):
        print("detecting...")
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame)
        self.result = self.landmarker.detect(mp_image)
        print(self.result)
        print("ok")

    def _detect_async(self, frame):
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame)
        self.landmarker.detect_async(
            mp_image,
            int(time.time()*1000))

    def has_result(self):
        return hasattr(self.result, "hand_landmarks") and \
            len(self.result.hand_world_landmarks)

    def get_hands_boundaries(self):
        boxes = []
        if not self.has_result(): return []
        for hand in self.result.hand_landmarks:
            box = _get_bounding_box(hand)
            if box == None: continue
            boxes.append(box)
        return boxes

    def close(self):
        self.landmarker.close()

