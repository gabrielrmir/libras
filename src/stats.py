import pandas as pd
from options import dataset_path

def main():
    df = pd.read_csv(dataset_path, header=None)
    print(df[0].value_counts())

if __name__ == '__main__':
    main()
