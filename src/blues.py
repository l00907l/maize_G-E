import pandas as pd
import numpy as np
from statsmodels.formula.api import ols
import statsmodels.formula.api as smf


# Load data
data = pd.read_csv('data/Training_Data/1_Training_Trait_Data_2014_2023.csv')

# Define environments
envs2016 = ['ARH1_2016', 'ARH2_2016', 'DEH1_2016', 'GAH1_2016', 'GAH2_2016', 'IAH1_2016', 
            'IAH2_2016', 'IAH3_2016', 'IAH4_2016', 'ILH1_2016', 'INH1_2016', 'KSH1_2016', 
            'MIH1_2016', 'MNH1_2016', 'MOH1_2016', 'NCH1_2016', 'NEH1_2016', 'NEH4_2016', 
            'NYH1_2016', 'NYH2_2016', 'OHH1_2016', 'ONH1_2016', 'ONH2_2016', 'TXH1_2016',
            'TXH2_2016', 'WIH1_2016', 'WIH2_2016']
envs2017 = ['ARH1_2017', 'ARH2_2017', 'COH1_2017', 'DEH1_2017', 'GAH1_2017', 'GAH2_2017', 
            'IAH1_2017', 'IAH2_2017', 'IAH3_2017', 'IAH4_2017', 'ILH1_2017', 'INH1_2017',
            'MIH1_2017', 'MNH1_2017', 'MOH1_2017', 'NCH1_2017', 'NEH3_2017', 'NEH4_2017', 
            'NYH1_2017', 'NYH2_2017', 'NYH3_2017', 'OHH1_2017', 'ONH1_2017', 'ONH2_2017',
            'TXH1-Dry_2017', 'TXH1-Early_2017', 'TXH1-Late_2017', 'TXH2_2017', 'WIH1_2017', 'WIH2_2017']      
envs2018 = ['ARH1_2018', 'ARH2_2018', 'DEH1_2018', 'GAH1_2018', 'GAH2_2018', 'IAH1_2018', 'IAH2_2018',
            'IAH3_2018', 'IAH4_2018', 'ILH1_2018', 'INH1_2018', 'KSH1_2018', 'MIH1_2018', 'MNH1_2018',
            'MOH1_1_2018', 'MOH1_2_2018', 'NCH1_2018', 'NEH2_2018', 'NYH1_2018', 'NYH2_2018','NYH3_2018', 
            'OHH1_2018', 'SCH1_2018', 'TXH1-Dry_2018', 'TXH1-Early_2018', 'TXH1-Late_2018', 'TXH2_2018', 'WIH1_2018', 'WIH2_2018']
envs2019 = ['DEH1_2019', 'GAH1_2019', 'GAH2_2019', 'GEH1_2019', 'IAH1_2019', 'IAH2_2019', 'IAH3_2019', 'IAH4_2019', 'ILH1_2019', 'INH1_2019', 
            'MIH1_2019', 'MNH1_2019', 'MOH1_2019', 'NCH1_2019', 'NEH1_2019', 'NEH2_2019', 'NYH1_2019', 'NYH2_2019', 'NYH3_2019', 'OHH1_2019',
            'ONH2_2019', 'SCH1_2019', 'TXH1_2019', 'TXH2_2019', 'TXH3_2019', 'TXH4_2019', 'WIH1_2019', 'WIH2_2019']
envs2020 = ['DEH1_2020', 'GAH1_2020', 'GAH2_2020', 'GEH1_2020', 'IAH1_2020', 'INH1_2020', 'MIH1_2020', 'MNH1_2020', 'MOH1_1_2020', 'MOH1_2_2020', 'NCH1_2020', 'NEH1_2020', 'NEH2_2020', 
            'NEH3_2020', 'NYH2_2020', 'NYH3_2020', 'NYS1_2020', 'OHH1_2020', 'SCH1_2020', 'TXH1_2020', 'TXH2_2020', 'TXH3_2020', 'WIH1_2020', 'WIH2_2020', 'WIH3_2020']
envs2021 = ['COH1_2021', 'DEH1_2021', 'GAH1_2021', 'GAH2_2021', 'GEH1_2021', 'IAH1_2021', 'IAH2_2021', 'IAH3_2021', 'IAH4_2021', 'ILH1_2021',
            'INH1_2021', 'MIH1_2021', 'MNH1_2021', 'NCH1_2021', 'NEH1_2021', 'NEH2_2021', 'NEH3_2021', 'NYH2_2021', 'NYH3_2021', 'NYS1_2021',
            'SCH1_2021', 'TXH1_2021', 'TXH2_2021', 'TXH3_2021', 'WIH1_2021', 'WIH2_2021', 'WIH3_2021']
envs2022 = ['COH1_2022', 'DEH1_2022', 'GAH1_2022', 'GAH2_2022', 'GEH1_2022', 'IAH1_2022', 'IAH2_2022', 'IAH3_2022', 'IAH4_2022', 'ILH1_2022', 'INH1_2022', 'MIH1_2022', 
            'MNH1_2022', 'MOH2_2022', 'NCH1_2022', 'NEH1_2022', 'NEH2_2022', 'NEH3_2022', 'NYH2_2022', 'NYH3_2022', 'OHH1_2022', 'SCH1_2022', 'TXH1_2022', 'TXH2_2022', 
            'TXH3_2022', 'WIH1_2022', 'WIH2_2022', 'WIH3_2022']
envs2023 = ['COH1_2023', 'DEH1_2023', 'GAH1_2023', 'GAH2_2023', 'IAH1_2023', 'IAH2_2023', 'IAH3_2023', 'IAH4_2023', 'ILH1_2023', 'INH1_2023', 'MIH1_2023', 'MNH1_2023', 
            'MOH1_2023', 'MOH2_2023', 'NCH1_2023', 'NEH1_2023', 'NEH3_2023', 'NYH2_2023', 'NYH3_2023', 'OHH1_2023', 'SCH1_2023', 'TXH1_2023', 'TXH2_2023', 'TXH3_2023', 
            'WIH1_2023', 'WIH2_2023', 'WIH3_2023']


envs = envs2016 + envs2017 + envs2018 + envs2019 + envs2020 + envs2021 + envs2022 + envs2023

# Filter data for the defined environments
data = data[data['Env'].isin(envs)]
data = data.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 23]]
data = data.drop(columns=['Field_Location'])
data = data[data['Hybrid'] != 'LOCAL_CHECK']
data = data.sort_values(by='Experiment')
data = data.reset_index(drop=True)
data['rep'] = data['Replicate'].astype(str) + ":" + data['Block'].astype(str)
for variable in ['Env', 'Experiment', 'Replicate', 'Block', 'rep', 'Plot', 'Range', 'Pass', 'Hybrid']:
    data[variable] = data[variable].astype('category')


blues = pd.DataFrame()
cvs_h2s = pd.DataFrame()

for env in envs:
    print(f"Processing environment: {env}")
    
    # 提取当前环境的数据 删除没有产量的数据（类似 R 的 droplevels）
    data_env = data[data['Env'] == env].copy()
    data_env = data_env.dropna(subset=['Yield_Mg_ha']).reset_index(drop=True)
    
    # 初始化固定效应和随机效应
    fixed_blues = 'Yield_Mg_ha ~ C(Hybrid) + C(Replicate)'
    fixed_h2 = 'Yield_Mg_ha ~ C(Replicate)'
    random = ['C(Replicate):C(Block)', 'C(Range)', 'C(Pass)']
    random_h2 = ['C(Hybrid)', 'C(Replicate):C(Block)', 'C(Range)', 'C(Pass)']
    
    # 动态调整随机效应
    if data_env['Range'].isna().all():
        print('Removing Range factor')
        random = [r for r in random if 'C(Range)' not in r]
        random_h2 = [r for r in random_h2 if 'C(Range)' not in r]
    
    if data_env['Pass'].isna().all():
        print('Removing Pass factor')
        random = [r for r in random if 'C(Pass)' not in r]
        random_h2 = [r for r in random_h2 if 'C(Pass)' not in r]
    
    if len(data_env['Block'].unique()) == 1:
        print('Removing nesting Block factor')
        random = ['C(Replicate)' if r == 'C(Replicate):C(Block)' else r for r in random]
        random_h2 = ['C(Replicate)' if r == 'C(Replicate):C(Block)' else r for r in random_h2]
    
    if env == 'WIH1_2021':
        print('Removing Range due to singularity with block')
        random = [r for r in random if 'C(Range)' not in r]
        random_h2 = [r for r in random_h2 if 'C(Range)' not in r]
    
    # （1）计算 BLUEs
    # 使用固定效应公式拟合 OLS 模型
    formula_blues = fixed_blues
    model_blues = smf.ols(formula_blues, data=data_env).fit()
    
    # 提取 Hybrid 的系数（BLUEs）
    hybrid_coefs = model_blues.params.filter(like='C(Hybrid)')
    # BLUEs预测值
    predictions = model_blues.predict(data_env)
    blue_data = pd.DataFrame({'Env': env, 'Hybrid': data_env['Hybrid'], 'predicted_value': predictions})
    blues = blues.append(blue_data, ignore_index=True)

    # （2）计算 CV
    residual_variance = np.var(model_blues.resid)
    mean_yield = data_env['Yield_Mg_ha'].mean()
    cv = np.sqrt(residual_variance) / mean_yield
    
    # （3）计算遗传力（H2）
    # 使用固定效应公式拟合 H2 模型
    formula_h2 = fixed_h2
    model_h2 = smf.ols(formula_h2, data=data_env).fit()
    
    # 估计 Hybrid 方差（模拟混合模型中的随机效应）
    hybrid_variance = np.var(hybrid_coefs.values)
    h2 = hybrid_variance / (hybrid_variance + residual_variance)
    
    # 保存 CV 和 H2
    cvs_h2s_df = pd.DataFrame([{'Env': env, 'cv': cv, 'h2': h2}])
    cvs_h2s = pd.concat([cvs_h2s, cvs_h2s_df], ignore_index=True)
    print(f"Environment: {env}, CV: {cv}, H2: {h2}")

# 转换为 pandas 数据框
blues_df = pd.DataFrame(blues)
cvs_h2s_df = pd.DataFrame(cvs_h2s)

# 保存为 CSV 文件
blues_df.to_csv('output/2016/blues.csv', index=False)
cvs_h2s_df.to_csv('output/2016/cvs_h2s.csv', index=False)

print("Results saved:")
print("BLUES saved to 'output/2016/blues.csv'")
print("CVs and H2s saved to 'output/2016/cvs_h2s.csv'")