import numpy as np
import math
from pathlib import Path

def hand_to_2d_flipped_array(hand, flip = 0):
    x_scale = 1.0
    if flip:
        x_scale = -1.0
    arr = np.zeros((len(hand),2))
    for i in range(len(hand)):
        arr[i][0] = hand[i].x*x_scale
        arr[i][1] = hand[i].y
    return arr

def hand_to_3d_array(hand):
    arr = np.zeros((len(hand),3))
    for i in range(len(hand)):
        arr[i][0] = hand[i].x
        arr[i][1] = hand[i].y
        arr[i][2] = hand[i].z
    return arr

# Cálculos vetoriais

def vec_len(np_arr):
    return math.sqrt((np_arr**2).sum())

def vec_norm(np_arr):
    l = vec_len(np_arr)
    if l == 0:
        return np.zeros(np_arr.shape)
    return np_arr/l

# Array circular de vetores

class CArray():
    def __init__(self, shape):
        self._arr = np.zeros(shape)
        self._front = 0
        self._size = shape[0]

    def push(self, elem):
        self._arr[self._front] = elem
        self._front = (self._front+1)%self._size

    def avg(self):
        return self._arr.sum(axis=0)/self._size

def is_image(path: Path):
    suffix = path.suffix
    if suffix == '.png': return True
    if suffix == '.jpg': return True
    return False

def list_dirs(path: Path):
    return [x for x in path.iterdir() if x.is_dir()]

def list_images(path: Path):
    return [x for x in path.iterdir() if is_image(x)]
