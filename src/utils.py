import numpy as np
import math

def hand_to_2d_array(hand):
    arr = np.zeros((len(hand),2))
    for i in range(len(hand)):
        arr[i][0] = hand[i].x
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
        np.zeros(np_arr.shape)
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
