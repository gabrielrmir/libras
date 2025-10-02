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

import pandas as pd

dirs = ['data/asl_dataset']
target = 'data/dataset.csv'

def PrepareDir(dir):
    return pd.DataFrame()

def Prepare(dirs, target = None):
    data = []
    for dir in dirs:
        data.append(PrepareDir(dir))
    # TODO: join dataframes here
    # TODO: dataframe.to_csv(target)

if __name__ == '__main__':
    Prepare(dirs, target)
