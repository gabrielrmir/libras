import cv2
import numpy as np
import typing

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import utils
from dataset import load_dataset
from camera import Camera
from landmarker import Landmarker
from options import dataset_path, classifier_algorithm

def _load_clf(option):
    match option:
        case 'knn':
            return KNeighborsClassifier(weights='distance', n_neighbors=11)
        case 'randomforest':
            return RandomForestClassifier()
        case _:
            print("[ERROR]: invalid classifier algorithm")
            exit(1)

class Classifier():
    def __init__(self, dataset_path):
        X, y = load_dataset(dataset_path)
        self.clf = _load_clf(classifier_algorithm)
        self.clf.fit(X,y)

    def predict(self, X):
        return self.clf.predict(X)

def draw_motion_line(frame, pt1, dir):
    cv2.line(frame, pt1, pt1+dir, (255,0,0), 5)

def main():
    cv2.namedWindow("Classifier", cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
    classifier = Classifier(dataset_path)
    landmarker = Landmarker()
    cam = Camera()
    cap = cam.cap
    center: typing.Any
    center = np.array(cam.size, dtype=int)//2

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        key = cv2.waitKey(1)
        if key == 27: # esc
            break

        label = ''
        landmarker.detect(frame)

        is_moving = False
        if not landmarker.result.is_empty():
            rect = landmarker.get_hand_rect()
            utils.draw_rect(frame, rect)

            dir = landmarker.global_motion.avg()
            if utils.vec_len(dir) > .1:
                is_moving = True

            dir = (dir[:2]*100).astype(int)
            draw_motion_line(frame, center, dir)

            x = 150
            for m in landmarker.local_motion:
                motion = (m.avg()[:2]*100).astype(int)
                draw_motion_line(frame, center+np.array([x, -100]), motion)
                x -= 75

            hand = utils.hand_to_2d_array(landmarker.result.world.get_hand()).flatten()
            y = classifier.predict([hand])[0]
            label = str(y)

        cv2.flip(frame, 1, frame)
        utils.draw_text(frame, label, (10,40))
        utils.draw_text(frame, 'moving:'+str(is_moving), (10,80))
        cv2.imshow('Classifier', frame)

    cap.release()

if __name__ == '__main__':
    main()
