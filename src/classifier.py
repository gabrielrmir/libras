import cv2
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score

import utils
from dataset import load_dataset
from camera import Camera
from landmarker import Landmarker
from options import dataset_path

# X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
# accuracy = accuracy_score(y_test, y_pred)
# print(accuracy)

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
        if landmarker.has_result():
            hand = landmarker.result.hand_world_landmarks[0]
            hand = utils.hand_to_1d_array(hand)
            box = landmarker.get_hands_boundaries()[0]
            utils.draw_box(frame, box)

            dir = landmarker.global_motion.avg()
            l = utils.vec_len(dir)
            print(l)
            if l > .1: is_moving = True

            utils.vec_len(dir)
            line_end = (center+dir[:2]*100).astype(int).tolist()
            cv2.line(frame, center.tolist(), line_end, (255,0,0), 5)
            label = str(classifier.predict([hand]))

        cv2.flip(frame, 1, frame)
        utils.draw_text(frame, label, (10,40))
        utils.draw_text(frame, 'moving:'+str(is_moving), (10,80))
        cv2.imshow('Classifier', frame)

    cap.release()
