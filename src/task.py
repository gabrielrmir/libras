import cv2

from landmarker import Landmarker
from camera import Camera
import options

class Task():
    def __init__(self, title):
        self.cam = Camera()
        self.scale = options.display_scale
        self.dsize = (int(self.cam.size[0]*self.scale), int(self.cam.size[1]*self.scale))
        self.cap = self.cam.cap
        self.title = title
        cv2.namedWindow(self.title,
            cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
        cv2.moveWindow(self.title, 10, 10)
        self.landmarker = Landmarker(options.running_mode)
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
        frame = cv2.resize(frame, self.dsize)
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
