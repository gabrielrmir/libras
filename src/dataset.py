import pandas as pd
import utils

class Dataset():
    def __init__(self, filename):
        self.filename = filename

    def save(self, label, hand):
        l = utils.hand_to_1d_array(hand)
        if l == None:
            print('could not save hand to dataset')
            return

        l.insert(0, label)
        pd.DataFrame([l]).to_csv(self.filename, mode='a', header=False, index=False)
        print('saved hand as label: ' + label)
