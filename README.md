# Projeto Sistema de Tradução de Movimento

## Uso

Para executar o programa de captura:

```script
python ./src/capture.py
```

## Dependências

Recomenda-se utilizar Python 3.12 com as seguintes dependências instaladas

- [TensorFlow](https://www.tensorflow.org/)
- [OpenCV-python](https://github.com/opencv/opencv-python)
- [Mediapipe](https://github.com/google-ai-edge/mediapipe)

Tais dependências podem ser instaladas através do seguinte comando:

```script
pip install opencv-python mediapipe tensorflow
```

Também é necessário ter o modelo `hand_landmarker.task` instalado no diretório `models/`. Para isso, execute o comando `make` ou baixe [manualmente](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker).

