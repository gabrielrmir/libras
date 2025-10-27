from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from dataset import load_dataset
from options import dataset_path, classifier_algorithm
from task import Task
import utils
import draw

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

# def draw_motion_line(frame, pt1, dir):
#     cv2.line(frame, pt1, pt1+dir, (255,0,0), 5)

class ClassifierTask(Task):
    def __init__(self):
        super().__init__('Classifier')
        self.classifier = Classifier(dataset_path)

    def _process(self, frame):
        self.landmarker.detect(frame)
        if self.landmarker.result.is_empty():
            return

        hand = self.landmarker.result.get_hand()
        hand = (utils.hand_to_2d_array(hand)*self.cam.size).astype(int)
        draw.hand_box(frame, hand)

        is_moving = False
        # dir = landmarker.global_motion.avg()
        # if utils.vec_len(dir) > .1:
        #     is_moving = True

        # dir = (dir[:2]*100).astype(int)
        # draw_motion_line(frame, center, dir)

        # x = 150
        # for m in landmarker.local_motion:
        #     motion = (m.avg()[:2]*100).astype(int)
        #     draw_motion_line(frame, center+np.array([x, -100]), motion)
        #     x -= 75

        hand = utils.hand_to_2d_array(
            self.landmarker.result.world.get_hand()).flatten()
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
