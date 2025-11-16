# Projeto Sistema de Tradução de Libras

# Dependências

Para executar o programa é necessário ter as seguintes dependências instaladas:

* Python 3.12

Criar ambiente virtual:

```script
python -m venv venv
```

Carregar ambiente virtual:

```script
source venv/bin/activate
```

Instalar dependências:

```script
pip install -r requirements.txt
```

Também é necessário ter o modelo `hand_landmarker.task` instalado no diretório `data/`. Para isso, execute o comando `make` ou baixe [manualmente](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task).

# Uso

Para executar o programa:

    python ./src/main.py [opções] [modo]

Opções:

    [modo]:
    capture                         Ferramenta de captura usada para construção
                                    do dataset.
    classifier [knn|randomforest]   Ferramenta de classificação de gestos,
                                    depende do dataset criado pela ferramenta
                                    de captura; Pode ser informada o algoritmo
                                    classificador, usa KNN por padrão; É a
                                    ferramenta padrão quando o modo é omitido.
    stats                           Ferramenta de estatística; Informa a
                                    quantidade de capturas para cada gesto.
    [opções]:
    --model <path>                  Caminho para o arquivo com o modelo do hand
                                    landmarker; Caso omitido, usa o caminho em
                                    src/options.py.
    --dataset <path>                Caminho para o arquivo contendo os dados de
                                    captura dos gestos; Caso omitido, usa o
                                    caminho em src/options.py.

