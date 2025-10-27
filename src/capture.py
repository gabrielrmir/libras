from dataset import Dataset
from task import Task

from options import dataset_path
import utils
import draw

class CaptureTask(Task):
    def __init__(self):
        super().__init__('Libras')
        self.label = 'a'
        self.dataset = Dataset(dataset_path)

    def _input(self, key):
        if key == ord('l'):
            self.label = input('label>')
        elif key == ord('c') and \
            not self.landmarker.result.is_empty():
            self.dataset.save(self.label,
                self.landmarker.result.world.get_hand())

            # p = Path(capture_dir, label)
            # p.mkdir(mode=0o755, parents=True, exist_ok=True)
            # filename = uuid.uuid4().hex + '.jpg'
            # cv2.imwrite(str(p / filename), cropped_im)

    def _process(self, frame):
        draw.text(frame, f'Label: {self.label}', (10,40))

        self.landmarker.detect(frame)
        if self.landmarker.result.is_empty():
            return

        hand = self.landmarker.result.get_hand()
        hand = (utils.hand_to_2d_array(hand)*self.cam.size).astype(int)

        draw.hand_box(frame, hand)
        draw.hand_lines(frame, hand)
        draw.hand_dots(frame, hand)

def main():
    task = CaptureTask()
    while task.running:
        task.update()

if __name__ == '__main__':
    main()
