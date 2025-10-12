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

def _get_bounding_box(hand):
    hand = utils.hand_to_2d_array(hand)
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
    def __init__(self, mode: RunningMode = RunningMode.LIVE_STREAM):
        self.result = HandLandmarkerResult
        # self.last_result = HandLandmarkerResult

        self.landmarker = self.create_landmarker(mode)
        self.running_mode = mode
        self.detect = self._detect_async if mode == RunningMode.LIVE_STREAM else self._detect_sync

        # Movimento relativo acumulado para a ponta dos dedos (4, 8, 12, 16, 20)
        # TODO: Implementar movimentação local
        self.local_motion = np.ndarray((5,2))

        # == Movimento relativo da mão ==
        # O movimento é calculado em relação ao ponto 0
        # Obs.: Talvez seja interessante utilizar o centróide [0,5,17] ao invés
        # de um único ponto
        self.global_motion = CArray((5,3))

    def _detect_callback(self, result, *_):
        had_result = self.has_result()
        last_result = self.result
        self.result = result

        if not had_result or not self.has_result():
            return

        hand = self.result.hand_landmarks[0]
        last_hand = last_result.hand_landmarks[0]
        p0_last = np.array([last_hand[0].x,last_hand[0].y,last_hand[0].z])
        p0 = np.array([hand[0].x,hand[0].y,hand[0].z])
        p1 = np.array([hand[1].x,hand[1].y,hand[1].z])

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

