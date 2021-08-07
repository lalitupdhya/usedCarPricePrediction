from sklearn import ensemble

MODELS = {
    "random_forest" : ensemble.RandomForestRegressor(n_estimators=600, max_depth=10,
                                                        max_features='log2',
                                                        max_leaf_nodes=500,
                                                        n_jobs=-1, verbose=2, random_state=142),
    "extra_trees" : ensemble.ExtraTreesRegressor(n_estimators=600, n_jobs=-1, verbose=2)
}