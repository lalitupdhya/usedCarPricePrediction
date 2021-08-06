from sklearn import ensemble

MODELS = {
    "random_forest" : ensemble.RandomForestRegressor(n_estimators=200, n_jobs=-1, verbose=2),
    "extra_trees" : ensemble.ExtraTreesRegressor(n_estimators=200, n_jobs=-1, verbose=2)
}