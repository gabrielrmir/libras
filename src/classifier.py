import cv2
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

import utils
from dataset import load_dataset
from camera import Camera
from landmarker import Landmarker
from options import dataset_path

class Classifier():
    def __init__(self, dataset_path):
        X, y = load_dataset(dataset_path)
        self.clf = Pipeline(steps=[
            # Os dados provenientes do MediaPipe já são normalizados, esta etapa não é necessária
            # ("scaler", StandardScaler()),

            # Diferentes algoritmos
            # TODO: adicionar option para qual tipo de classificador usar + option para n_neighbors
            ("knn", KNeighborsClassifier(weights='distance', n_neighbors=11))
            # ("random_forest", RandomForestClassifier())
        ])
        self.clf.fit(X,y)

    def predict(self, X):
        return self.clf.predict(X)

if __name__ == '__main__':
    cv2.namedWindow("Classifier", cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
    classifier = Classifier(dataset_path)
    landmarker = Landmarker()
    cam = Camera()
    cap = cam.cap
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

            utils.vec_len(dir)
            line_end = (center+dir[:2]*100).astype(int).tolist()
            cv2.line(frame, center.tolist(), line_end, (255,0,0), 5)

            hand = utils.hand_to_2d_array(landmarker.result.world.get_hand()).flatten()
            y = classifier.predict([hand])[0]
            label = str(y)

        cv2.flip(frame, 1, frame)
        utils.draw_text(frame, label, (10,40))
        utils.draw_text(frame, 'moving:'+str(is_moving), (10,80))
        cv2.imshow('Classifier', frame)

    cap.release()
