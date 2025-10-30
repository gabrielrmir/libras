import cv2
import numpy as np

# Mapeamento das conexões entre coordenadas das mãos
LINES = ((0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),
    (7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),
    (14,15),(15,16),(13,17),(0,17),(17,18),(18,19),(19,20))

FONT = cv2.FONT_HERSHEY_PLAIN

COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)

COLOR_TEXT = COLOR_GREEN

def arrow(frame, pt1, pt2, color, pointer_color, thickness):
    cv2.line(frame, pt1, pt2, color, thickness)
    cv2.circle(frame, pt2, 8, pointer_color, -1)

def hand_dots(frame, hand):
    for pos in hand:
        cv2.circle(frame, pos, 4, COLOR_RED, -1)

def hand_lines(frame, hand):
    for line in LINES:
        cv2.line(frame, hand[line[0]],
            hand[line[1]], COLOR_BLUE, 2)

def hand_box(frame, hand):
    cx = hand[:,0]
    cy = hand[:,1]
    cv2.rectangle(frame,
        (cx.min(), cy.min()),
        (cx.max(), cy.max()),
        COLOR_GREEN, 1)

def motion_2d(frame, hand_2d, motion, scale = 100):
    pt1 = np.mean(hand_2d[motion.sources], axis=0).astype(int)
    offset = (motion.get_motion()[:2]*scale).astype(int)
    arrow(frame, pt1, pt1+offset, COLOR_BLUE, COLOR_RED, 4)

def text_box(frame, text, pos):
    (size, baseline) = cv2.getTextSize(text, FONT, fontScale=3, thickness=1)
    cv2.rectangle(frame, (pos[0], pos[1]+baseline), (pos[0]+size[0],pos[1]-size[1]-baseline), (0,0,0), -1)
    cv2.putText(frame, text, pos, FONT, 3, (255,255,255), 2, cv2.LINE_AA | cv2.LINE_8, False)

def text(frame, text, pos):
    cv2.putText(frame, text, pos, 0, 1, COLOR_TEXT, 2)

def texts(frame, texts, pos):
    x,y = pos
    for text in texts:
        cv2.putText(frame, text, (x,y), 0, 1, COLOR_TEXT, 2)
        y += 32
