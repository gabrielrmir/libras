# Prepara as imagens para a planilha de dados em csv

# A pasta deve conter pastar nomeadas de acordo com o rótulo que representa cada uma das imagens
# dir/
# |--a/
# |  |--img1.jpg
# |  |--img2.jpg
# |  |--img3.jpg
# |
# |--b/
# |  |--img1.jpg
# |  |--img2.jpg
# |  |--img3.jpg
# |
# |--c/
#    |--img1.jpg
#    |--img2.jpg
#    |--img3.jpg

from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode as RunningMode
import pandas as pd
import landmarker
import cv2
import os
from pathlib import Path
import utils

dirs = ['data/asl_dataset/asl_dataset']
target = 'data/dataset.csv'
font = cv2.FONT_HERSHEY_SIMPLEX

def Prepare(dirs, target = None):
    lm = landmarker.Landmarker(RunningMode.IMAGE)
    dataframe = pd.DataFrame()
    key = cv2.waitKey(1)
    for dir in dirs:
        for label in os.listdir(dir):
            for img in os.listdir(Path(dir, label)):
                frame = cv2.imread(str(Path(dir, label, img)))
                lm.detect(frame)
                if lm.has_result():
                    utils.draw_hand(frame, lm.result.hand_landmarks[0])
                    cv2.text
                else:
                    cv2.putText(frame, 'NOT DETECTED', (10,60), font, 1, (0,0,255), 2)
                    continue
                cv2.putText(frame, 'label: '+label, (10,30), font, 1, (255,255,255), 2)
                cv2.imshow('prepare', frame)

                while True:
                    key = cv2.waitKey(1)
                    if key == ord('n'):
                        break
                    if key == ord('q'):
                        exit(0)
    # TODO: join dataframes here
    # TODO: dataframe.to_csv(target)

if __name__ == '__main__':
    Prepare(dirs, target)
