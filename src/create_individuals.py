from pathlib import Path

import pandas as pd


OUTPUT_PATH = Path('output')


ytrain = pd.read_csv("output/2016/cv/ytrain_seed6.csv")
yval = pd.read_csv("output/2016/cv/yval_seed6.csv")
test = pd.read_csv("output/2016/cv/test_seed6.csv")

train_hybrids = ytrain['Hybrid'].unique()
val_hybrids = yval['Hybrid'].unique()
test_hybrids = test['Hybrid'].unique()

all_hybrids = pd.Series(list(set(train_hybrids) | set(val_hybrids) | set(test_hybrids)))


all_hybrids.to_csv(OUTPUT_PATH / 'individuals.csv', index=False, header=False)