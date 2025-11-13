from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode as RunningMode
from landmarker import Landmarker
import utils
from pathlib import Path
import conv
from dataset import Dataset
import options

def main():
    lm = Landmarker(RunningMode.IMAGE)

    dataset_path = Path(options.dataset_path)
    dt = Dataset(dataset_path)

    train_path = Path('train')
    train_dirs = utils.list_dirs(train_path)
    for train_dir in train_dirs:
        label = train_dir.parts[-1]
        image_paths = utils.list_images(train_dir)
        image_paths = image_paths[:100]

        hands = [conv.filepath_to_coords(image_path, lm) for image_path in image_paths]
        hands = [c for c in hands if c is not None]

        # dataset_path.parent.mkdir(parents=True, exist_ok=True)
        for hand in hands:
            dt.save(label, hand)

if __name__ == '__main__':
    main()
