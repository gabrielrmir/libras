from functools import reduce
from mediapipe.tasks.python.components.containers.landmark import Landmark
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerResult
import mediapipe as mp
import time

class Landmarker():
    def __init__(self):
        self.result = HandLandmarkerResult
        self.landmarker =  mp.tasks.vision.HandLandmarker
        self.create_landmarker()

    def create_landmarker(self):
        def callback(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
            self.result = result

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path='./models/hand_landmarker.task'),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            result_callback=callback)

        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def detect(self, frame):
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
        if not self.has_result(): return []
        tl = Landmark(0,0)
        for hand in self.result.hand_landmarks:
            tl = reduce(lambda a, b: Landmark(min(a.x,b.x), min(a.y,b.y)), hand)
        return tl

    def close(self):
        self.landmarker.close()

