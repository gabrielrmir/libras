import cv2
import numpy as np
import math

# Conexões entre os pontos da mão

lines = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(0,17),(17,18),(18,19),(19,20)]

# Desenhos

def draw_hand_lines(frame, hand):
    h, w, _ = frame.shape

    for line in lines:
        x1,y1 = (int(hand[line[0]].x*w),int(hand[line[0]].y*h))
        x2,y2 = (int(hand[line[1]].x*w),int(hand[line[1]].y*h))
        cv2.line(frame,(x1,y1),(x2,y2),(255,0,0),2)

def draw_box(frame, box, pad = 0.0):
    x1,y1,x2,y2 = box
    h, w, _ = frame.shape
    pt1 = (int(x1*w-pad), int(y1*h-pad))
    pt2 = (int(x2*w+pad), int(y2*h+pad))
    cv2.rectangle(frame, pt1, pt2, (0,255,0), 2)

def draw_hand(frame, hand):
    h, w, _ = frame.shape
    draw_hand_lines(frame,hand)
    for pos in hand:
        x,y = (int(pos.x*w),int(pos.y*h))
        cv2.circle(frame,(x,y), 8, (0,0,255), -1)

def draw_hands(frame, landmarker):
    hands = landmarker.result.hand_landmarks
    for hand in hands:
        draw_hand(frame,hand)

def draw_text(frame, text, pos, color = (0,0,0)):
    cv2.putText(frame, text, pos, 0, 1, color, 2)

def hand_to_1d_array(hand):
    arr = []
    for pos in hand:
        if pos.x == None or pos.y == None: return None
        arr.append(pos.x)
        arr.append(pos.y)
    return arr

def hand_to_2d_array(hand):
    arr = []
    for pos in hand:
        if pos.x == None or pos.y == None: return None
        arr.append((pos.x, pos.y))
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
