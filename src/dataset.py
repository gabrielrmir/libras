import csv

class Dataset():
    def __init__(self, filename):
        self._file = open(filename, 'a')
        self._csvwriter = csv.writer(self._file)

    def save(self, symbol, hand):
        data = [symbol]
        for pos in hand:
            data.append(pos.x)
            data.append(pos.y)
        self._csvwriter.writerow(data)
        self._file.flush()

    def save_image(self, image):
        pass

    def close(self):
        self._file.close()
