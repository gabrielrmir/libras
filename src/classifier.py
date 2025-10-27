from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import time

from dataset import load_dataset
import draw
from options import dataset_path, classifier_algorithm, refresh_time
from task import Task

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

class ClassifierTask(Task):
    def __init__(self):
        super().__init__('Classifier')
        self.classifier = Classifier(dataset_path)

    def _process(self, frame):
        if time.time()-self.landmarker.timestamp > refresh_time:
            self.landmarker.detect(frame)

        hand = (self.landmarker.result[:,:2]*self.cam.size).astype(int)
        draw.hand_box(frame, hand)

        is_moving = False

        hand = self.landmarker.world_result[:,:2].flatten()
        y = self.classifier.predict([hand])[0]
        label = str(y)

        draw.texts(frame, [
            f'Label: {label}',
            f'Moving: {is_moving}',
        ], (10,40))

def main():
    task = ClassifierTask()
    while task.running:
        task.update()

if __name__ == '__main__':
    main()
