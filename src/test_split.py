from dataset import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
import options
import time
import matplotlib.pyplot as plt

def main():
    X, y = load_dataset(options.dataset_path)
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=.8)
    print(f'X_train = {len(X_train)}, y_train = {len(y_train)}')
    print(f'X_test = {len(X_test)}, y_test = {len(y_test)}')

    knn_clf = KNeighborsClassifier(options.knn_n_neighbors)
    knn_clf.fit(X_train, y_train)
    before = time.time()
    knn_pred = knn_clf.predict(X_test)
    knn_score = knn_clf.score(X_test, y_test)
    knn_time = time.time()-before

    rf_clf = RandomForestClassifier()
    rf_clf.fit(X_train, y_train)
    before = time.time()
    rf_pred = rf_clf.predict(X_test)
    rf_score = rf_clf.score(X_test, y_test)
    rf_time = time.time()-before

    print('========== knn ==========')
    ConfusionMatrixDisplay.from_predictions(y_test, knn_pred, xticks_rotation='vertical', colorbar=False)
    print(f'knn: {knn_score} {knn_time}')
    plt.show()

    print('===== random forest =====')
    ConfusionMatrixDisplay.from_predictions(y_test, rf_pred, xticks_rotation='vertical', colorbar=False)
    print(f'random forest: {rf_score} {rf_time}')
    plt.show()

if __name__ == '__main__':
    main()
