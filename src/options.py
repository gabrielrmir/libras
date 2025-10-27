# Caminho para o modelo de reconhecimento de esqueleto das mãos
# Pode ser baixado usando o makefile ou através do seguinte link:
# https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
landmarker_model_path = './data/hand_landmarker.task'

# Arquivo de destino para a ferramenta de captura
# Este arquivo também é usado para treinar o modelo de classificação de gestos
dataset_path = "./data/capture.csv"

# Opções:
# knn, randomforest
classifier_algorithm = "knn"
