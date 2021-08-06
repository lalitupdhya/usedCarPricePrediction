import os
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import metrics
import pandas as pd
import numpy as np
from . import dispatcher

TRAINING_DATA = os.environ["TRAINING_DATA"]
FOLD = int(os.environ["FOLD"])
MODEL = os.environ["MODEL"]
FOLD_MAPPING = {
    0: [1,2,3,4],
    1: [0,2,3,4],
    2: [0,1,3,4],
    3: [0,1,2,4],
    4: [0,1,2,3]
}

if __name__ == "__main__":

    df = pd.read_csv(TRAINING_DATA)

    train_df = df[df.kfold.isin(FOLD_MAPPING[FOLD])]
    valid_df = df[df.kfold==FOLD]

    y_train = train_df.Price.values
    y_valid = valid_df.Price.values

    train_df = train_df.drop(['Price', 'kfold'], axis=1)
    valid_df = valid_df.drop(['Price', 'kfold'], axis=1)

    valid_df = valid_df[train_df.columns]

    label_encoders = []
    for col in train_df.columns:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(train_df[col].values.tolist() + valid_df[col].values.tolist())
        train_df.loc[:, col] = lbl.transform(train_df[col].values.tolist())
        valid_df.loc[:, col] = lbl.transform(valid_df[col].values.tolist())

        label_encoders.append((col, lbl))

    #data is ready to train

    reg = dispatcher.MODELS[MODEL]
    reg.fit(train_df, y_train)

    preds = reg.predict(valid_df)
    print(metrics.mean_squared_log_error(y_valid, preds))