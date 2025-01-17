from math import floor

import pandas as pd
from sklearn.model_selection import GroupKFold


def create_field_location(df: pd.DataFrame):
    df['Field_Location'] = df['Env'].str[:-5]
    return df


def process_metadata(path: str, encoding: str = 'latin-1'):
    df = pd.read_csv(path, encoding=encoding)
    df['City'] = df['City'].str.strip().replace({'College Station, Texas': 'College Station'})
    df = df.rename(columns={
        'Weather_Station_Latitude (in decimal numbers NOT DMS)': 'weather_station_lat',
        'Weather_Station_Longitude (in decimal numbers NOT DMS)': 'weather_station_lon'
    })
    df['treatment_not_standard'] = (df['Treatment'] != 'Standard').astype('int')
    return df


def process_test_data(path: str):
    df = pd.read_csv(path)
    df = create_field_location(df)
    return df


def lat_lon_to_bin(x, step: float):
    if pd.notnull(x):
        return floor(x / step) * step
    else:
        return x


def agg_yield(df: pd.DataFrame):
    df['Year'] = df['Env'].str[-4:].astype('int')
    df_agg = (
        df
        .groupby(['Env', 'Hybrid'])
        .agg(
            weather_station_lat=('weather_station_lat', 'mean'),
            weather_station_lon=('weather_station_lon', 'mean'),
            treatment_not_standard=('treatment_not_standard', 'mean'),
            Field_Location=('Field_Location', 'last'),
            Year=('Year', 'last'),
            Yield_Mg_ha=('Yield_Mg_ha', 'mean')  # unadjusted means per env/hybrid combination
        )
        .reset_index()
    )
    return df_agg


def process_blues(df: pd.DataFrame):
    df['predicted_value'] = df.apply(lambda x: x['Yield_Mg_ha'] if x['predicted_value'] < 0 else x['predicted_value'], axis=1)
    df = df.drop('Yield_Mg_ha', axis=1)
    df = df.rename(columns={'predicted_value': 'Yield_Mg_ha'})
    return df


def feat_eng_weather(df: pd.DataFrame):
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
    df['month'] = df['Date'].dt.month 
    df['season'] = df['month'] % 12 // 3 + 1  # https://stackoverflow.com/a/44124490/11122513
    df['season'] = df['season'].map({1: 'winter', 2: 'spring', 3: 'summer', 4: 'fall'})
    df_agg = df.dropna(subset=[x for x in df.columns if x not in ['Env', 'Date']]).copy()
    df_agg = (
        df
        .groupby(['Env', 'season'])
        .agg(
            T2M_max=('T2M', 'max'),
            T2M_min=('T2M', 'min'),
            T2M_std=('T2M', 'std'),
            T2M_mean=('T2M', 'mean'),

            T2M_MIN_max=('T2M_MIN', 'max'),
            T2M_MIN_std=('T2M_MIN', 'std'),
            T2M_MIN_cv=('T2M_MIN', lambda x: x.std() / x.mean()),

            WS2M_max=('WS2M', 'max'),

            RH2M_max=('RH2M', 'max'),
            RH2M_p90=('RH2M', lambda x: x.quantile(0.9)),

            QV2M_mean=('QV2M', 'mean'),

            PRECTOTCORR_max=('PRECTOTCORR', 'max'),
            PRECTOTCORR_median=('PRECTOTCORR', 'median'),
            PRECTOTCORR_n_days_less_10_mm=('PRECTOTCORR', lambda x: sum(x < 10)),

            ALLSKY_SFC_PAR_TOT_std=('ALLSKY_SFC_PAR_TOT', 'std'),

        )
        .reset_index()
        .pivot(index='Env', columns='season')
    )
    df_agg.columns = ['_'.join(col) for col in df_agg.columns]
    return df_agg


def feat_eng_soil(df: pd.DataFrame):
    df_agg = (
        df
        .groupby('Env')
        .agg(
            Nitrate_N_ppm_N=('Nitrate-N ppm N', 'mean'),
            lbs_N_A=('lbs N/A', 'mean'),
            percentage_Ca_Sat=('%Ca Sat', 'mean')
        )
    )
    return df_agg


# def feat_eng_target(df: pd.DataFrame, ref_year: list, lag: int):
#     assert lag >= 1
#     features_list = []
    
#     for year in ref_year:
#         df_year = df[df['Year'] <= year - lag]       
#         col = f'yield_lag_{lag}_ref_{year}'
#         df_agg = (
#             df_year
#             .groupby('Field_Location')
#             .agg(
#                 **{f'mean_{col}': ('Yield_Mg_ha', 'mean')},
#                 **{f'min_{col}': ('Yield_Mg_ha', 'min')},
#                 **{f'p1_{col}': ('Yield_Mg_ha', lambda x: x.quantile(0.01))},
#                 **{f'q1_{col}': ('Yield_Mg_ha', lambda x: x.quantile(0.25))},
#                 **{f'q3_{col}': ('Yield_Mg_ha', lambda x: x.quantile(0.75))},
#                 **{f'p90_{col}': ('Yield_Mg_ha', lambda x: x.quantile(0.90))},
#             )
#         )       
#         features_list.append(df_agg)
    
#     df_final = features_list[0]
#     for df_agg in features_list[1:]:
#         df_final = df_final.merge(df_agg, on='Field_Location', how='left')

#     return df_final


def feat_eng_target(df: pd.DataFrame, lag: int = 1):
    '''
    自动生成滞后期特征。根据数据中的年份自动选择参考年份。
    Parameters:
    - df: 输入的 DataFrame,必须包含 'Year', 'Yield_Mg_ha', 和 'Field_Location' 列。
    - lag: 滞后期，默认为 1。
    Returns:
    - df_final: 包含滞后期特征的 DataFrame。
    '''
    assert lag >= 1, "滞后期 lag 必须大于等于 1"

    ref_year = df['Year'].unique()
    features_list = []
    
    for year in ref_year:
        df_year = df[df['Year'] <= year - lag]
        col = f'yield_lag_{lag}_ref_{year}'
        df_agg = (
            df_year
            .groupby('Field_Location')
            .agg(
                **{f'mean_{col}': ('Yield_Mg_ha', 'mean')},
                **{f'min_{col}': ('Yield_Mg_ha', 'min')},
                **{f'p1_{col}': ('Yield_Mg_ha', lambda x: x.quantile(0.01))},
                **{f'q1_{col}': ('Yield_Mg_ha', lambda x: x.quantile(0.25))},
                **{f'q3_{col}': ('Yield_Mg_ha', lambda x: x.quantile(0.75))},
                **{f'p90_{col}': ('Yield_Mg_ha', lambda x: x.quantile(0.90))},
            )
        )       
        features_list.append(df_agg)
    
    df_final = features_list[0]
    for df_agg in features_list[1:]:
        df_final = df_final.merge(df_agg, on='Field_Location', how='left')

    return df_final
         

def extract_target(df: pd.DataFrame):
    y = df['Yield_Mg_ha']
    del df['Yield_Mg_ha']
    return y


# def create_folds(df: pd.DataFrame, val_year: int, fillna: bool, random_state: int):
#     '''
#     Targets with NA are due to discarded plots (accordingly with Cyverse data)
#     Reference for CVs: "Genome-enabled Prediction Accuracies Increased by Modeling Genotype x Environment Interaction in Durum Wheat" (Sukumaran et. al, 2017)
#     https://acsess.onlinelibrary.wiley.com/doi/10.3835/plantgenome2017.12.0112
#     '''

#     if fillna:
#         raise NotImplementedError('"fillna" is not implemented.')
    
#     train = df[df['Year'].isin([2017,2018,2019,2020,2021,2022])].dropna(subset=['Yield_Mg_ha'])
#     val = df[df['Year'] == val_year].dropna(subset=['Yield_Mg_ha'])
#     train = train.reset_index(drop=True)
#     val = val.reset_index(drop=True)
#     train = train.sample(frac=1, random_state=random_state).reset_index(drop=True)
#     val = val.sample(frac=1, random_state=random_state).reset_index(drop=True)
#     df = pd.concat([train, val], axis=0, ignore_index=True)

#     return df

def create_folds(train_df: pd.DataFrame, test_df: pd.DataFrame, random_state: int):
    train_df = train_df[train_df['Year'] >= 2016]
    test_df['env_clean']= test_df['Env'].apply(lambda x: x[:4])
    test_env_distribution = test_df['env_clean'].value_counts(normalize=True)

    val_samples = []
    for env, proportion in test_env_distribution.items():
        if env=='ONH3': env='ONH2'
        env_data = train_df[train_df['Field_Location'] == env]
        n_samples = int(proportion * len(test_df))
        if len(env_data) >= n_samples:
            sampled_env_data = env_data.sample(n=n_samples, random_state=random_state)
            val_samples.append(sampled_env_data)
        else:
            print(f"Not enough samples for environment {env}, skipping.")
            exit()

    val_df = pd.concat(val_samples, axis=0, ignore_index=True)

    val_gen_env = val_df[['Hybrid', 'Env']].drop_duplicates()
    train_df = train_df.merge(val_gen_env, on=['Hybrid', 'Env'], how='left', indicator=True)
    train_df = train_df[train_df['_merge'] == 'left_only'].drop(columns=['_merge'])
    
    # Shuffle train and validation sets
    train = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    val = val_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return train, val

def create_folds_all(train_df: pd.DataFrame, random_state: int):
    train = train_df[train_df['Year'] >= 2016].dropna(subset=['Yield_Mg_ha'])
    train = train.reset_index(drop=True)
    train = train.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return train