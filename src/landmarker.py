from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerResult
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode as RunningMode
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerOptions
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker
import mediapipe as mp
import time
import utils
from utils import CArray
import numpy as np
from options import landmarker_model_path, refresh_time

def _is_empty(result: HandLandmarkerResult | None):
    return (not result) or \
        (not hasattr(result, "hand_landmarks")) or \
        len(result.hand_landmarks) == 0

class Landmarker():
    def __init__(self, mode: RunningMode = RunningMode.LIVE_STREAM):
        self.result = np.zeros((21,3))
        self.world_result = np.zeros((21,3))
        self.timestamp = 0

        self.prev_result = np.zeros((21,3))
        self.prev_world_result = np.zeros((21,3))
        self.prev_timestamp = 0

        self.running_mode = mode
        self.landmarker = self.create_landmarker(self.running_mode)

        self.detect = self._detect_async
        if mode != RunningMode.LIVE_STREAM:
            self.detect = self._detect_sync

        # Movimento relativo acumulado para a ponta dos dedos (4, 8, 12, 16, 20)
        self.local_index = (4, 8, 12, 16, 20)
        self.local_motion = [
            CArray((5,3)),
            CArray((5,3)),
            CArray((5,3)),
            CArray((5,3)),
            CArray((5,3))
        ]

        # == Movimento relativo da mão ==
        # O movimento é calculado em relação ao ponto 0
        # Obs.: Talvez seja interessante utilizar o centróide [0,5,17] ao invés
        # de um único ponto
        self.global_motion = CArray((5,3))

    def _detect_callback(self, result: HandLandmarkerResult, timestamp_sec):
        if _is_empty(result):
            return

        self.prev_result = self.result
        self.prev_world_result = self.world_result
        self.prev_timestamp = self.timestamp

        self.result = utils.hand_to_3d_array(result.hand_landmarks[0])
        self.world_result = utils.hand_to_3d_array(result.hand_world_landmarks[0])
        self.timestamp = timestamp_sec

        # == Cálculo de movimentação ==
        # O vetor de movimento deve ser convertido para uma escala onde a
        # distância entre os pontos 0 e 1 deve ser sempre igual à 1. Isso ajuda
        # a garantir uma movimentação independente da profundidade da mão.

        p0 = np.array(self.result[0])
        p1 = np.array(self.result[1])
        scale = utils.vec_len(p1-p0)

        pc_last = (self.prev_result[0]+self.prev_result[5]+self.prev_result[17])/3
        pc = (self.result[0]+self.result[5]+self.result[17])/3
        centroid_motion = pc-pc_last
        self.global_motion.push(centroid_motion/scale)

        for i in range(len(self.local_index)):
            j = self.local_index[i]
            self.local_motion[i].push((self.result[j]-self.prev_result[j])/scale)

    def create_landmarker(self, mode: RunningMode):
        callback = None
        if mode == RunningMode.LIVE_STREAM:
            def async_callback(result, _, timestamp_ms):
                self._detect_callback(result, timestamp_ms/1000)
            callback = async_callback

        options = HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=landmarker_model_path),
            running_mode=mode,
            result_callback=callback)

        return HandLandmarker.create_from_options(options)

    def _detect_sync(self, frame):
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame)
        self._detect_callback(self.landmarker.detect(mp_image), time.time())

    def _detect_async(self, frame):
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame)
        self.landmarker.detect_async(
            mp_image,
            int(time.time()*1000))

    def close(self):
        self.landmarker.close()


