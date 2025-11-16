import time

from dataset import Dataset
import draw
from options import dataset_path, refresh_time
from task import Task

class CaptureTask(Task):
    def __init__(self):
        super().__init__('Libras')
        self.label = 'a'
        self.dataset = Dataset(dataset_path)
        self.frozen = False
        self.counter = 0

    def _input(self, key):
        if key == ord('l'):
            self.label = input('label>')
        elif key == ord('c'):
            self.dataset.save(self.label, self.landmarker.world_result[:,:2])
            self.counter += 1
            print(f'Contador: {self.counter}')
        elif key == ord('f'):
            self.frozen = not self.frozen

            # p = Path(capture_dir, label)
            # p.mkdir(mode=0o755, parents=True, exist_ok=True)
            # filename = uuid.uuid4().hex + '.jpg'
            # cv2.imwrite(str(p / filename), cropped_im)

    def _process(self, frame):
        if not self.frozen and time.time()-self.landmarker.timestamp > refresh_time:
            self.landmarker.detect(frame)

        hand = (self.landmarker.result[:,:2]*self.cam.size).astype(int)
        draw.hand_box(frame, hand)
        draw.hand_lines(frame, hand)
        draw.hand_dots(frame, hand)

        draw.texts(frame, [
            f'Label: {self.label}',
            f'{'Frozen' if self.frozen else ''}'
        ], (10,40))

def main(label = 'a'):
    task = CaptureTask()
    task.label = label
    while task.running:
        task.update()

if __name__ == '__main__':
    main()
