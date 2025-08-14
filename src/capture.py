import csv
import cv2
from landmarker import Landmarker
import utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit(1)

landmarker = Landmarker()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Unable to receive frame. Exiting...")
        break

    landmarker.detect_async(frame)

    if hasattr(landmarker.result, "hand_landmarks"):
        w,h = 640,360
        hands = landmarker.result.hand_landmarks
        for hand in hands:
            utils.draw_hand(frame,hand,(w,h))

        hands = landmarker.result.hand_world_landmarks
        for hand in hands:
            utils.draw_hand_lines(frame,hand,(w,h),(w//8,h//8))


    cv2.imshow('Libras', frame)
    if cv2.waitKey(1) == ord('q'):
        break

landmarker.close()
cap.release()
cv2.destroyAllWindows()
