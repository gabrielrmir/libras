import mediapipe as mp
import time

class Landmarker():
    def __init__(self):
        self.result = mp.tasks.vision.HandLandmarkerResult
        self.landmarker =  mp.tasks.vision.HandLandmarker
        self.create_landmarker()

    def create_landmarker(self):
        def callback(result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
            self.result = result

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path='./models/hand_landmarker.task'),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            result_callback=callback)

        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def detect_async(self, frame):
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame)
        self.landmarker.detect_async(
            mp_image,
            int(time.time()*1000))

    def close(self):
        self.landmarker.close()

