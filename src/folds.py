# creating folds for train
from sklearn import model_selection
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

if __name__== "__main__":
    path = './usedCarData/clean_data.csv'
    df = pd.read_csv(path)

    df["kfold"] = -1
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    kfold = model_selection.StratifiedKFold(n_splits=5)
    for fold, (train_idx, val_idx) in enumerate(kfold.split(df, df.Price)):
        df.loc[val_idx, "kfold"] = fold
    
    df.to_csv('./usedCarData/train_folds.csv', index=False)
