import sys
import options

import classifier
import capture
import statistics

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
            case _:
                _err_invalid()

def _cmd_capture(args):
    capture.main()

def _cmd_classifier(args):
    if len(args):
        c = args.pop(0)
        if c == 'knn' or c == 'randomforest':
            options.classifier_algorithm = c
        else:
            _err_invalid()
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
                                    src/options.py.""")

def _cmd_stats():
    statistics.main()

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
        case _:
            _err_invalid()

if __name__ == '__main__':
    main()
