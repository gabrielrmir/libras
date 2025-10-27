import cv2

# Mapeamento das conexões entre coordenadas das mãos
LINES = ((0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),
    (7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),
    (14,15),(15,16),(13,17),(0,17),(17,18),(18,19),(19,20))

text_color = (0,255,0)

def hand_dots(frame, hand):
    for pos in hand:
        cv2.circle(frame, pos, 4, (0,0,255), -1)

def hand_lines(frame, hand):
    for line in LINES:
        cv2.line(frame, hand[line[0]],
            hand[line[1]], (255,0,0), 2)

def hand_box(frame, hand):
    cx = hand[:,0]
    cy = hand[:,1]
    cv2.rectangle(frame,
        (cx.min(), cy.min()),
        (cx.max(), cy.max()),
        (0,255,0), 2)

# TODO: Reimplementar funções de visualização de movimento
def motion_local(): pass
def motion_global(): pass

def text(frame, text, pos):
    cv2.putText(frame, text, pos, 0, 1, text_color, 2)

def texts(frame, texts, pos):
    x,y = pos
    for text in texts:
        cv2.putText(frame, text, (x,y), 0, 1, text_color, 2)
        y += 32
