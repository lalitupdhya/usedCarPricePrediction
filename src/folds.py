from sklearn import model_selection
import pandas as pd
import numpy as np

path = './usedCarData/clean_data.csv'

df = pd.read_csv(path)

print(df.head())