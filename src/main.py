import sys
import options

def _err_invalid():
    print("[ERROR]: Invalid usage")
    print("Check help for more information.")
    exit(1)

def parse_flags(args):
    while len(args):
        if not args[0].startswith('--'):
            break
        flag = args.pop(0)[2:]
        match flag:
            case 'dataset':
                if len(args) == 0:
                    _err_invalid()
                path = args.pop(0)
                options.dataset_path = path
            case 'model':
                if len(args) == 0:
                    _err_invalid()
                path = args.pop(0)
                options.landmarker_model_path = path
            case 'refresh':
                if len(args) == 0:
                    _err_invalid()
                try:
                    refresh_time = float(args.pop(0))
                    options.refresh_time = refresh_time
                except:
                    _err_invalid()
            case _:
                _err_invalid()

def _cmd_capture(args):
    label = 'a'
    if len(args):
        label = args.pop(0)
    import capture
    capture.main(label)

def _cmd_classifier(args):
    if len(args):
        c = args.pop(0)
        if c == 'knn' or c == 'randomforest':
            options.classifier_algorithm = c
        else:
            _err_invalid()
    import classifier
    classifier.main()

def _cmd_help():
    print("""Uso: python ./src/main.py [opções] [modo]

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
    --dataset <path>                Caminho para o arquivo contendo os dados de
                                    captura dos gestos; Caso omitido, usa o
                                    caminho em src/options.py.
    --model <path>                  Caminho para o arquivo com o modelo do hand
                                    landmarker; Caso omitido, usa o caminho em
                                    src/options.py.
    --refresh <secs>                Taxa de atualização dos landmarks em
                                    segundos; Números maiores podem melhorar a
                                    latência; Valor padrão: 0.01""")

def _cmd_stats():
    import stats
    stats.main()

# Obter informações de acurácia de um conjunto de imagens de teste
# As imagens devem ser dispostas no diretório 'test' onde cada subdiretório
# corresponde ao label previsto. É importante que o label esteja em letra
# minúscula. O nome das imagens pode ser completamente arbitrário.
# Ex.:
# test/a/test001.jpg
# test/a/test002.jpg
# test/a/test003.jpg
# ...
# test/b/test001.jpg
# test/b/test002.jpg
# test/b/test003.jpg
# ...
def _cmd_accuracy():
    import accuracy
    accuracy.main()

# Ferramenta de captura alternativa. Captura região da câmera e salva como
# imagem. Pode ser informado o label da pose que está sendo capturada
def _cmd_crop(args):
    label = 'a'
    if len(args):
        label = args.pop(0)
    import crop
    crop.main(label)

# Processamento de imagens de treino para dataset em csv
# Dados processados serão acrescentados no arquivo informado pela opção
# options.dataset_path
# As imagens seguem a mesma estrutura informada no comando de acurácia, usando
# o diretório 'train' no lugar de 'test'.
def _cmd_process():
    pass

def main():
    args = sys.argv[1:]

    parse_flags(args)

    command = 'classifier'
    if len(args):
        command = args.pop(0)

    match command:
        case 'capture':
            _cmd_capture(args)
        case 'classifier':
            _cmd_classifier(args)
        case 'help':
            _cmd_help()
        case 'stats':
            _cmd_stats()
        case 'accuracy':
            _cmd_accuracy()
        case 'crop':
            _cmd_crop(args)
        case 'process':
            _cmd_process()
        case _:
            _err_invalid()

if __name__ == '__main__':
    main()
