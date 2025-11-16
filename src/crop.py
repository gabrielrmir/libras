from camera import Camera
import cv2
import numpy as np
import draw
from pathlib import Path
import time

def crop(image, pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    return image[y1:y2, x1:x2]

def save(label, image):
    name = str(int(time.time()*1000))
    output_path = Path('train') / label / f'{name}.jpg'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # cv2.imshow(label, image)
    cv2.imwrite(output_path, image)

def main(label = 'a'):
    cam = Camera()
    cap = cam.cap
    size = np.array((256, 256))
    tl = np.array(cam.center-size/2)-np.array((0, 64))
    pt1 = tl.astype(int).tolist()
    pt2 = (tl+size).astype(int).tolist()
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        key = cv2.waitKey(1)
        if key == ord('c'):
            im = crop(frame, pt1, pt2)
            save(label, im)
            count += 1
            print(f'contador: {count}')
        if key == ord('q'):
            break

        cv2.rectangle(frame, pt1, pt2, (0,0,255), 2)
        draw.text_box(frame, f'capture: {label}', (0,38))
        cv2.imshow('crop', frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
