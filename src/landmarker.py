# from mediapipe.tasks.python.components.containers.landmark import Landmark

from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerResult
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode as RunningMode
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerOptions
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker
import mediapipe as mp
import time
import utils
from utils import CArray
import numpy as np
from options import landmarker_model_path

def _is_empty(result: HandLandmarkerResult | None):
    return (not result) or \
        (not hasattr(result, "hand_landmarks")) or \
        len(result.hand_landmarks) == 0

class Result():
    class _world():
        def __init__(self, res):
            self._parent: Result
            self._parent = res

        def get_hand(self):
            assert(self._parent._result != None)
            return self._parent._result.hand_world_landmarks[0]

        def __getitem__(self, key: int) -> tuple[float, float, float]:
            assert(self._parent._result != None)
            pos = self._parent._result.hand_world_landmarks[0][key]
            return (pos.x, pos.y, pos.z)
        
        def __len__(self):
            if self._parent.is_empty():
                return 0
            assert(self._parent._result != None)
            return len(self._parent._result.hand_landmarks[0])

    def __init__(self, result = None):
        self._result: HandLandmarkerResult | None
        self._result = result
        self.world = self._world(self)
        self._empty = _is_empty(result)

    def get_hand(self):
        assert(self._result != None)
        return self._result.hand_landmarks[0]

    def set(self, new_result):
        self._result = new_result
        self._empty = _is_empty(new_result)

    def __getitem__(self, key: int) -> tuple[float, float, float]:
        assert(self._result != None)
        pos = self._result.hand_landmarks[0][key]
        return (pos.x, pos.y, pos.z)

    def __len__(self):
        if self.is_empty():
            return 0
        assert(self._result != None)
        return len(self._result.hand_landmarks[0])

    def is_empty(self):
        return self._empty

class Landmarker():
    def __init__(self, mode: RunningMode = RunningMode.LIVE_STREAM):
        self.result: Result
        self.result = Result()

        self._prev_result: Result
        self._prev_result = Result()

        self.landmarker = self.create_landmarker(mode)
        self.running_mode = mode
        self.detect = self._detect_async
        if mode != RunningMode.LIVE_STREAM:
            self.detect = self._detect_sync

        # Movimento relativo acumulado para a ponta dos dedos (4, 8, 12, 16, 20)
        # TODO: Implementar movimentação local
        self.local_motion = np.ndarray((5,2))

        # == Movimento relativo da mão ==
        # O movimento é calculado em relação ao ponto 0
        # Obs.: Talvez seja interessante utilizar o centróide [0,5,17] ao invés
        # de um único ponto
        self.global_motion = CArray((5,3))

    def _detect_callback(self, result, *_):
        self._prev_result.set(self.result._result)
        self.result.set(result)

        if self._prev_result.is_empty() or \
            self.result.is_empty():
            return

        p0_last = np.array(self._prev_result[0])
        p0 = np.array(self.result[0])
        p1 = np.array(self.result[1])

        # == Cálculo de movimentação ==
        # O vetor de movimento deve ser convertido para uma escala onde a
        # distância entre os pontos 0 e 1 deve ser sempre igual à 1. Isso ajuda
        # a garantir uma movimentação independente da profundidade da mão.
        motion = p0-p0_last
        l = utils.vec_len(motion)/utils.vec_len(p1-p0)
        self.global_motion.push(utils.vec_norm(motion)*l)

    def create_landmarker(self, mode: RunningMode):
        callback = None
        if mode == RunningMode.LIVE_STREAM:
            callback = self._detect_callback

        options = HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=landmarker_model_path),
            running_mode=mode,
            result_callback=callback)

        return HandLandmarker.create_from_options(options)

    def _detect_sync(self, frame):
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame)
        self._detect_callback(self.landmarker.detect(mp_image))

    def _detect_async(self, frame):
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame)
        self.landmarker.detect_async(
            mp_image,
            int(time.time()*1000))

    def get_hand_rect(self):
        assert(not self.result.is_empty())

        p0 = self.result[0]
        rect = [p0[0], p0[1], p0[0], p0[1]]

        for i in range(1, len(self.result)):
            pos = self.result[i]
            rect[0] = min(rect[0], pos[0])
            rect[1] = min(rect[1], pos[1])
            rect[2] = max(rect[2], pos[0])
            rect[3] = max(rect[3], pos[1])

        return rect

    def close(self):
        self.landmarker.close()

