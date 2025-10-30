from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import time

from dataset import load_dataset
import draw
import options
from task import Task
from history import History

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
        self.clf = _load_clf(options.classifier_algorithm)
        self.clf.fit(X,y)

    def predict(self, X):
        return self.clf.predict(X)

class ClassifierTask(Task):
    def __init__(self):
        super().__init__('Classifier')
        self.classifier = Classifier(options.dataset_path)
        self.history = History()
        self.last_result = 0.0

    def _process(self, frame):
        current_time = time.time()
        landmarker_delta = current_time-self.landmarker.timestamp
        if landmarker_delta > options.refresh_time:
            self.landmarker.detect(frame)
        if landmarker_delta > options.reset_timeout:
            self.history.clear()

        hand = (self.landmarker.result[:,:2]*self.cam.size).astype(int)
        draw.hand_box(frame, hand)
        for motion in self.landmarker.motions:
            draw.motion_2d(frame, hand, motion)

        label = ''
        if self.landmarker.timestamp > self.last_result:
            self.last_result = self.landmarker.timestamp
            hand = self.landmarker.world_result[:,:2].flatten()
            y = self.classifier.predict([hand])[0]
            label = str(y)

            if label:
                self.history.push_label(label, current_time)

        draw.text_box(frame, str(self.history), (0,38))

def main():
    task = ClassifierTask()
    while task.running:
        task.update()

if __name__ == '__main__':
    main()
