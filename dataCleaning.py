import pandas as pd
import numpy as np
import re

path_train = './usedCarData/train.csv'

df_train = pd.read_csv(path_train)

df_train.columns = ['ID', 'Price', 'Levy', 'Manufacturer', 'Model', 'ProdYear',
                    'Category', 'LeatherInterior', 'FuelType', 'EngineVolume', 'Mileage',
                    'Cylinders', 'GearBoxType', 'DriveWheels', 'Doors', 'Wheel', 'Color',
                    'Airbags']

df_train.Levy = df_train.Levy.replace('-',np.nan).astype('float')
df_train.ProdYear = 2021 - df_train.ProdYear
df_train.Mileage = df_train.Mileage.replace(' km', '', regex=True).astype(int)
df_train = df_train.drop('ID', axis=1)




