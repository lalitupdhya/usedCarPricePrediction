import pandas as pd
import numpy as np
from sklearn import impute
import re
from sklearn.experimental import enable_iterative_imputer
from sklearn import linear_model
from sklearn import tree
from sklearn import ensemble
from sklearn import neighbors
from sklearn import pipeline
from sklearn import model_selection
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')


path_train = './usedCarData/train.csv'

df_train = pd.read_csv(path_train)

df_train.columns = ['ID', 'Price', 'Levy', 'Manufacturer', 'Model', 'ProdYear',
                    'Category', 'LeatherInterior', 'FuelType', 'EngineVolume', 'Mileage',
                    'Cylinders', 'GearBoxType', 'DriveWheels', 'Doors', 'Wheel', 'Color',
                    'Airbags']

#converting data types and minor feature engineering
df_train.Levy = df_train.Levy.replace('-',np.nan).astype('float')
df_train.ProdYear = 2021 - df_train.ProdYear
df_train.Mileage = df_train.Mileage.replace(' km', '', regex=True).astype(int)
df_train = df_train.drop('ID', axis=1)

#creating a feature Turbo
condition = df_train.EngineVolume.str.contains('Turbo')
df_train['Turbo'] = np.where(condition, 1, 0)

df_train.EngineVolume = df_train.EngineVolume.replace(' Turbo','', regex=True).astype('float')


estimators = [
    linear_model.BayesianRidge(),
    tree.DecisionTreeRegressor(max_features='sqrt', random_state=0),
    ensemble.ExtraTreesRegressor(n_estimators=10, random_state=0),
    neighbors.KNeighborsRegressor(n_neighbors=15)]


score_iterative_imputer =pd.DataFrame()
br_estimator = linear_model.BayesianRidge()

columnX = ['Levy', 'ProdYear', 'EngineVolume', 'Turbo', 'Mileage', 'Cylinders',
          'Airbags','Turbo']
columnY = ['Price']

for estimator_imputer in estimators:
    estimator = pipeline.make_pipeline(impute.IterativeImputer(missing_values=np.nan,
                                                              estimator = estimator_imputer),
                                      br_estimator)
    score_iterative_imputer[estimator_imputer] = model_selection.cross_val_score(estimator,
                                                                                X = df_train[columnX], 
                                                                                y = df_train[columnY],
                                                                                scoring = 'neg_mean_squared_log_error',
                                                                                cv=5)   



