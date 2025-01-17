#!/bin/bash

# fit G+E models

python src/run_lightgbm.py
echo '[G+E] lightgbm model ok'

python src/run_xgb.py
echo '[G+E] xgboost model ok'
