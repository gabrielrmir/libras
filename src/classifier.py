import cv2

from sklearn.neighbors import KNeighborsClassifier
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
            # ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=11))
        ])
        self.clf.fit(X,y)
    def predict(self, X):
        return self.clf.predict(X)

if __name__ == '__main__':
    classifier = Classifier(dataset_path)
    landmarker = Landmarker()
    cam = Camera()
    cap = cam.cap

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        key = cv2.waitKey(1)
        if key == 27: # esc
            break

        label = ''
        landmarker.detect(frame)
        if landmarker.has_result():
            hand = landmarker.result.hand_world_landmarks[0]
            hand = utils.hand_to_1d_array(hand)
            box = landmarker.get_hands_boundaries()[0]
            utils.draw_box(frame, box)
            label = str(classifier.predict([hand]))

        cv2.flip(frame, 1, frame)
        utils.draw_text(frame, label, (10,40))
        cv2.imshow('Classifier', frame)

    cap.release()
