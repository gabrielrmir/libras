import cv2

from landmarker import Landmarker
from camera import Camera
from options import running_mode

class Task():
    def __init__(self, title):
        self.cam = Camera()
        self.cap = self.cam.cap
        self.title = title
        cv2.namedWindow(self.title,
            cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
        self.landmarker = Landmarker(running_mode)
        self.running = True
        self.exit_code = 0

    def update(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self._exit(1)
            return

        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'): # <esc> ou <q> para sair:
            self._exit(0)
            return
        self._input(key)
        self._process(frame)
        if key == ord('k'):
            cv2.imshow('captura', frame)
        cv2.imshow(self.title, frame)

    def _input(self, key):
        pass

    def _process(self, frame):
        pass

    def _exit(self, exit_code = 0):
        self.exit_code = exit_code
        self.running = False
        self.landmarker.close()
        self.cap.release()
        cv2.destroyAllWindows()
