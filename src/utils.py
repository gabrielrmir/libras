import cv2

lines = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(0,17),(17,18),(18,19),(19,20)]

def draw_hand_lines(frame, hand):
    h, w, _ = frame.shape

    for line in lines:
        x1,y1 = (int(hand[line[0]].x*w),int(hand[line[0]].y*h))
        x2,y2 = (int(hand[line[1]].x*w),int(hand[line[1]].y*h))
        cv2.line(frame,(x1,y1),(x2,y2),(255,0,0),2)

def draw_box(frame, box):
    x1,y1,x2,y2 = box
    h, w, _ = frame.shape
    pt1 = (int(x1*w), int(y1*h))
    pt2 = (int(x2*w), int(y2*h))
    cv2.rectangle(frame, pt1, pt2, (0,255,0), 2)

def draw_hand(frame, hand):
    h, w, _ = frame.shape

    draw_hand_lines(frame,hand)
    for pos in hand:
        x,y = (int(pos.x*w),int(pos.y*h))
        cv2.circle(frame,(x,y), 8, (0,0,255), -1)

def hand_to_1d_array(hand):
    arr = []
    for pos in hand:
        if pos.x == None or pos.y == None: return None
        arr.append(pos.x)
        arr.append(pos.y)
    return arr

def hand_to_points_array(hand):
    arr = []
    for pos in hand:
        if pos.x == None or pos.y == None: return None
        arr.append((pos.x, pos.y))
    return arr
