import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
data1=pd.read_csv("D:/yan/竞赛/农业比赛/Maize_GxE/maize_lig_xgb/output/2016/result/submission_G(A)_E_nolag.csv")
data2=pd.read_csv("D:/yan/竞赛/农业比赛/Maize_GxE/maize/output/2016/result/submission_G(A)_E_nolag.csv")
rmse = np.sqrt(mean_squared_error(data1['Yield_Mg_ha'], data2['Yield_Mg_ha']))
print(rmse)