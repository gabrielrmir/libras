import csv
import cv2
import landmarker

cap = cv2.VideoCapture(2)
if not cap.isOpened():
    print("Cannot open camera")
    exit(1)

landmarker = landmarker.Landmarker()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Unable to receive frame. Exiting...")
        break

    landmarker.detect_async(frame)

    cv2.imshow('Libras', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
