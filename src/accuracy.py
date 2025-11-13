from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode as RunningMode
import mediapipe as mp
from landmarker import Landmarker
from classifier import Classifier
import options
import utils
from pathlib import Path
import cv2

def is_image(path: Path):
    suffix = path.suffix
    if suffix == '.png': return True
    if suffix == '.jpg': return True
    return False

def list_test_dirs(path: Path):
    return [x for x in path.iterdir() if x.is_dir()]

def list_test_images(path: Path):
    return [x for x in path.iterdir() if is_image(x)]

def main():
    lm = Landmarker(RunningMode.IMAGE)
    classifier = Classifier(options.dataset_path)

    def path_to_coords(path: Path):
        image = cv2.imread(str(path))
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        result = lm.landmarker.detect(mp_image)
        if not result.hand_world_landmarks:
            return None
        return utils.hand_to_2d_flipped_array(result.hand_world_landmarks[0], 0).flatten()

    test_path = Path('test')
    test_dirs = list_test_dirs(test_path)
    for test_dir in test_dirs:
        label = test_dir.parts[-1]
        image_paths = list_test_images(test_dir)
        image_paths = image_paths[:100]

        coords = [path_to_coords(image_path) for image_path in image_paths]
        coords = [c for c in coords if c is not None]

        res = classifier.predict(coords)
        correct = 0
        for l in res:
            if l == label:
                correct += 1

        print(f'<{label}>: {100*correct/len(res)} [{correct}/{len(res)}]')

if __name__ == '__main__':
    main()
