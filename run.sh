export TRAINING_DATA=usedCarData/train_folds.csv
export TEST_DATA=usedCarData/clean_test.csv
export MODEL=$1

FOLD=0 python3 -m src.train
FOLD=1 python3 -m src.train
FOLD=2 python3 -m src.train
FOLD=3 python3 -m src.train
FOLD=4 python3 -m src.train