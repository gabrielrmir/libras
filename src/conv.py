from pathlib import Path
from landmarker import Landmarker
import mediapipe as mp
import cv2
import utils

def filepath_to_coords(filepath: Path, lm: Landmarker):
    image = cv2.imread(str(filepath))
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    result = lm.landmarker.detect(mp_image)
    if not result.hand_world_landmarks:
        return None
    return utils.hand_to_2d_flipped_array(result.hand_world_landmarks[0], 0).flatten()
