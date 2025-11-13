import mediapipe as mp
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode as RunningMode
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerOptions
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerResult
import numpy as np
import time
import cv2

import options
import utils
from utils import CArray

def _is_empty(result: HandLandmarkerResult | None):
    return (not result) or \
        (not hasattr(result, "hand_landmarks")) or \
        len(result.hand_landmarks) == 0

class Motion():
    def __init__(self, *sources):
        self.sources = np.array(sources)
        self.values = CArray((5,3))

    def update(self, result, prev_result, scale = 1.0):
        pos = result[self.sources].sum(axis=0)/len(self.sources)
        pos_prev = prev_result[self.sources].sum(axis=0)/len(self.sources)
        motion = (pos-pos_prev)/scale
        self.values.push(motion)

    # Vetor não normalizado
    def get_motion(self):
        return self.values.avg()

class Landmarker():
    def __init__(self, mode: RunningMode = RunningMode.LIVE_STREAM):
        self.result = np.zeros((21,3))
        self.world_result = np.zeros((21,2))
        self.timestamp = 0

        self.prev_result = np.zeros((21,3))
        self.prev_world_result = np.zeros((21,2))
        self.prev_timestamp = 0

        self.running_mode = mode
        self.landmarker = self.create_landmarker(self.running_mode)

        self.detect = self._detect_async
        if mode != RunningMode.LIVE_STREAM:
            self.detect = self._detect_sync

        self.motions = [
            # Centróide da palma da mão
            Motion(0, 5, 17),

            # Ponta dos dedos
            Motion(4),
            Motion(8),
            Motion(12),
            Motion(16),
            Motion(20)
        ]

        self.handedness = 'Right'

    def _detect_callback(self, result: HandLandmarkerResult, timestamp_sec):
        if _is_empty(result):
            return

        # handedness = result.handedness[0][0]
        # self.handedness = handedness.display_name
        # flip = handedness.index

        self.prev_result = self.result
        self.prev_world_result = self.world_result
        self.prev_timestamp = self.timestamp

        # if handedness.index:
        #     print(f'{time.time()} flip')

        self.result = utils.hand_to_3d_array(result.hand_landmarks[0])
        self.world_result = utils.hand_to_2d_flipped_array(result.hand_world_landmarks[0], 0)
        self.timestamp = timestamp_sec

        # Cálculo de movimentação para cada conjunto de pontos listados em
        # motion; É aplicada uma escala para reduzir os impactos da
        # profundidade no movimento entre frames.
        scale = utils.vec_len(self.result[1]-self.result[0])
        for motion in self.motions:
            motion.update(self.result, self.prev_result, scale)

        # TODO: Talvez seja preciso levar em consideração o delta entre frames;
        # Pode ser inserido como parâmetro no método motion.update.

    def create_landmarker(self, mode: RunningMode):
        callback = None
        if mode == RunningMode.LIVE_STREAM:
            def async_callback(result, _, timestamp_ms):
                self._detect_callback(result, timestamp_ms/1000)
            callback = async_callback

        lm_options = HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=options.landmarker_model_path),
            running_mode=mode,
            result_callback=callback)

        return HandLandmarker.create_from_options(lm_options)

    def _detect_sync(self, frame):
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self._detect_callback(self.landmarker.detect(mp_image), time.time())

    def _detect_async(self, frame):
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.landmarker.detect_async(
            mp_image,
            int(time.time()*1000))

    def close(self):
        self.landmarker.close()
