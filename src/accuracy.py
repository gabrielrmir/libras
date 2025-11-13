from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode as RunningMode
from landmarker import Landmarker
from classifier import Classifier
import options
import utils
from pathlib import Path
import conv

def main():
    lm = Landmarker(RunningMode.IMAGE)
    classifier = Classifier(options.dataset_path)

    test_path = Path('test')
    test_dirs = utils.list_dirs(test_path)
    for test_dir in test_dirs:
        label = test_dir.parts[-1]
        image_paths = utils.list_images(test_dir)
        image_paths = image_paths[:100]

        coords = [conv.filepath_to_coords(image_path, lm) for image_path in image_paths]
        coords = [c for c in coords if c is not None]

        res = classifier.predict(coords)
        correct = 0
        for l in res:
            if l == label:
                correct += 1

        print(f'<{label}>: {100*correct/len(res)} [{correct}/{len(res)}]')

if __name__ == '__main__':
    main()
