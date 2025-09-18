import cv2

lines = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(0,17),(17,18),(18,19),(19,20)]

def draw_hand_lines(frame, hand, scale=(1,1), offset=(0,0)):
    w,h = scale
    xoff,yoff = offset
    for line in lines:
        x1,y1 = (int(hand[line[0]].x*w)+xoff,int(hand[line[0]].y*h)+yoff)
        x2,y2 = (int(hand[line[1]].x*w)+xoff,int(hand[line[1]].y*h)+yoff)
        cv2.line(frame,(x1,y1),(x2,y2),(255,0,0),2)

def draw_box(frame, box):
    cv2.rectangle(frame, box[0,1], box[2,3], (0,255,0), 2)

def draw_hand(frame, hand, scale=(1,1), offset=(0,0)):
    w,h = scale
    xoff,yoff = offset

    draw_hand_lines(frame,hand,scale,offset)
    for pos in hand:
        x,y = (int(pos.x*w)+xoff,int(pos.y*h)+yoff)
        cv2.circle(frame,(x,y), 8, (0,0,255), -1)


