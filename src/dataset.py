import pandas as pd
import utils

class Dataset():
    def __init__(self, filename):
        self.filename = filename

    def save(self, label, hand):
        l = utils.hand_to_2d_array(hand).flatten().tolist()
        if l == None:
            print('could not save hand to dataset')
            return

        l.insert(0, label)
        pd.DataFrame([l]).to_csv(self.filename, mode='a', header=False, index=False)
        print('saved hand as label: ' + label)

def load_dataset(filename):
    df = pd.read_csv(filename, header=None)
    X = df.iloc[:,1:].to_numpy()
    # X = X.reshape((len(X), len(X[0])//2, 2))
    y = df.iloc[:,0].to_numpy()
    return (X, y)
