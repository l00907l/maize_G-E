import gc
import argparse
import numpy as np
from pathlib import Path

import pandas as pd
import lightgbm as lgbm
from sklearn.decomposition import TruncatedSVD

from preprocessing import process_test_data, create_field_location
from evaluate import create_df_eval, avg_rmse, feat_imp
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=6)
parser.add_argument('--A', action='store_true', default=True)
parser.add_argument('--D', action='store_true', default=False)
parser.add_argument('--E', action='store_true', default=True)
parser.add_argument('--svd', action='store_true', default=False)
parser.add_argument('--n_components', type=int, default=100)
parser.add_argument('--nolag_features', action='store_true', default=True)
args = parser.parse_args()
ma="G(A)_E_nolag"

OUTPUT_PATH = Path(f'output/2016/cv')
TEST_PATH = 'data/Testing_Data/1_Submission_Template_2024.csv'


print('Using G+E model.')



def preprocess_g(df, kinship, individuals: list):
    df.columns = [x[:len(x) // 2] for x in df.columns]  # fix duplicated column names
    df.index = df.columns
    df = df[df.index.isin(individuals)]  # filter rows
    df = df[[col for col in df.columns if col in individuals]]  # filter columns
    df.index.name = 'Hybrid'
    df.columns = [f'{x}_{kinship}' for x in df.columns]
    return df


def preprocess_kron(df, kinship):
    df[['Env', 'Hybrid']] = df['id'].str.split(':', expand=True)
    df = df.drop('id', axis=1).set_index(['Env', 'Hybrid'])
    df.columns = [f'{x}_{kinship}' for x in df.columns]
    # print(df.info(), '\n')
    # df[df.columns] = np.array(df.values, dtype=np.float32)  # downcast is too slow
    # print(df.info(), '\n')
    return df


def prepare_gxe(kinship):
    kron = pd.read_feather(OUTPUT_PATH / f'kronecker_{kinship}.arrow')
    kron = preprocess_kron(kron, kinship=kinship)
    return kron


if __name__ == '__main__':
    df_sub = process_test_data(TEST_PATH).reset_index()[['Env', 'Hybrid']]

    # 基因+环境
    if args.E:
        # 环境数据
        print('Using E matrix.')
        xtrain = pd.read_csv(OUTPUT_PATH / f'train_seed{args.seed}.csv')
        xval = pd.read_csv(OUTPUT_PATH / f'val_seed{args.seed}.csv')
        xtest = pd.read_csv(OUTPUT_PATH / 'xtest.csv')
        
        xtrain_info = pd.concat([xtrain.iloc[:, :2], xtrain.iloc[:, -1]], axis=1)
        xval_info = pd.concat([xval.iloc[:, :2], xval.iloc[:, -1]], axis=1)
        xtest_info = xtest.iloc[:, :2].copy()

        xtrain_features = xtrain.iloc[:, 2:-1]
        xval_features = xval.iloc[:, 2:-1]
        xtest_features = xtest.iloc[:, 2:].copy()

        scaler = StandardScaler()
        xtrain_scaled = scaler.fit_transform(xtrain_features)
        xval_scaled = scaler.transform(xval_features)
        xtest_scaled = scaler.transform(xtest_features)

        pca = PCA(n_components=0.95)
        xtrain_pca = pca.fit_transform(xtrain_scaled)
        xval_pca = pca.transform(xval_scaled)
        xtest_pca = pca.transform(xtest_scaled)

        xtrain_pca_df = pd.DataFrame(xtrain_pca, index=xtrain.index)
        xval_pca_df = pd.DataFrame(xval_pca, index=xval.index)
        xtest_pca_df = pd.DataFrame(xtest_pca, index=xtest.index)

        xtrain = pd.concat([xtrain_info, xtrain_pca_df], axis=1)
        xval = pd.concat([xval_info, xval_pca_df], axis=1)
        xtest = pd.concat([xtest_info, xtest_pca_df], axis=1)

        # 基因数据
        if args.A or args.D:
            individuals = xtrain['Hybrid'].unique().tolist() + xval['Hybrid'].unique().tolist() + xtest['Hybrid'].unique().tolist()
            individuals = list(dict.fromkeys(individuals))
            print('# unique individuals (including test set):', len(individuals))
            kinships = []
            kroneckers = []
            if args.A:
                print('Using A matrix.')
                A = pd.read_csv('output/kinship_additive.txt', sep='\t')
                A = preprocess_g(A, 'A', individuals)
                kinships.append(A)
            if args.D:
                print('Using D matrix.')
                D = pd.read_csv('output/kinship_dominant.txt', sep='\t')
                D = preprocess_g(D, 'D', individuals)
                kinships.append(D)

            K = pd.concat(kinships, axis=1)
            del kinships
        print(K.head())
        print(xtrain.head())
        xtrain = pd.merge(xtrain, K, on='Hybrid', how='left').dropna().set_index(['Env', 'Hybrid'])
        xval = pd.merge(xval, K, on='Hybrid', how='left').dropna().set_index(['Env', 'Hybrid'])
        xtest = pd.merge(xtest, K, on='Hybrid', how='left').set_index(['Env', 'Hybrid'])
        gc.collect()


    # 单独基因
    else:
        ytrain = pd.read_csv(OUTPUT_PATH / f'ytrain_seed{args.seed}.csv')
        yval = pd.read_csv(OUTPUT_PATH / f'yval_seed{args.seed}.csv')
        xtest = pd.read_csv(TEST_PATH)

        if args.A or args.D:
            individuals = ytrain['Hybrid'].unique().tolist() + yval['Hybrid'].unique().tolist() + xtest['Hybrid'].unique().tolist()
            individuals = list(dict.fromkeys(individuals))
            print('# unique individuals (including test set):', len(individuals))
            kinships = []
            kroneckers = []
            if args.A:
                print('Using A matrix.')
                A = pd.read_csv('output/kinship_additive.txt', sep='\t')
                A = preprocess_g(A, 'A', individuals)
                kinships.append(A)
            if args.D:
                print('Using D matrix.')
                D = pd.read_csv('output/kinship_dominant.txt', sep='\t')
                D = preprocess_g(D, 'D', individuals)
                kinships.append(D)

            K = pd.concat(kinships, axis=1)
            del kinships
        
        xtrain = pd.merge(ytrain, K, on='Hybrid', how='left').dropna().set_index(['Env', 'Hybrid'])
        xval = pd.merge(yval, K, on='Hybrid', how='left').dropna().set_index(['Env', 'Hybrid'])
        xtest = pd.merge(xtest, K, on='Hybrid', how='left').set_index(['Env', 'Hybrid'])
        gc.collect()

    ytrain = xtrain['Yield_Mg_ha']
    del xtrain['Yield_Mg_ha']
    yval = xval['Yield_Mg_ha']
    del xval['Yield_Mg_ha']
    gc.collect()


    print(xtrain.head())
    print(xtest.head())

    # run model
    if not args.svd:
        lag_cols = xtrain.filter(regex='_lag', axis=1).columns
        if args.nolag_features:
            if len(lag_cols) > 0:
                xtrain = xtrain.drop(lag_cols, axis=1)
                xval = xval.drop(lag_cols, axis=1)
                xtest = xtest.drop(lag_cols, axis=1)

        xtrain = xtrain.reset_index()
        xtrain = create_field_location(xtrain)
        xtrain['Field_Location'] = xtrain['Field_Location'].astype('category')
        xtrain = xtrain.set_index(['Env', 'Hybrid'])
        xval = xval.reset_index()
        xval = create_field_location(xval)
        xval['Field_Location'] = xval['Field_Location'].astype('category')
        xval = xval.set_index(['Env', 'Hybrid'])
        xtest = xtest.reset_index()
        xtest = create_field_location(xtest)
        xtest['Field_Location'] = xtest['Field_Location'].astype('category')
        xtest = xtest.set_index(['Env', 'Hybrid'])

        print('Using full set of features.')
        print('# Features:', xtrain.shape[1])


    else:
        lag_cols = xtrain.filter(regex='_lag', axis=1).columns
        xtrain_lag = xtrain[lag_cols]
        xval_lag = xval[lag_cols]
        xtest_lag = xtest[lag_cols]
        if len(lag_cols) > 0:
            xtrain = xtrain.drop(lag_cols, axis=1)
            xval = xval.drop(lag_cols, axis=1)
            xtest = xtest.drop(lag_cols, axis=1)
        cols = [x for x in xtrain.columns.tolist() if x not in ['Env', 'Hybrid']]

        print('Using svd.')
        print('# Components:', args.n_components)
        svd = TruncatedSVD(n_components=args.n_components, random_state=args.seed)
        svd.fit(xtrain[cols]) 
        print('Explained variance:', svd.explained_variance_ratio_.sum())

        svd_cols = [f'svd{i}' for i in range(args.n_components)]
        xtrain_svd = pd.DataFrame(svd.transform(xtrain[cols]), columns=svd_cols, index=xtrain[cols].index)
        xval_svd = pd.DataFrame(svd.transform(xval[cols]), columns=svd_cols, index=xval[cols].index)
        xtest_svd = pd.DataFrame(svd.transform(xtest[cols]), columns=svd_cols, index=xtest[cols].index)
        del svd
        gc.collect()

        if args.nolag_features:
            xtrain = xtrain_svd.copy()
            del xtrain_svd
            xval = xval_svd.copy()
            del xval_svd
            xtest = xtest_svd.copy()
            del xtest_svd
            gc.collect()
        else:
            xtrain = pd.concat([xtrain_svd, xtrain_lag], axis=1)
            del xtrain_svd
            xval = pd.concat([xval_svd, xval_lag], axis=1)
            del xval_svd
            xtest = pd.concat([xtest_svd, xtest_lag], axis=1)
            del xtest_svd
            gc.collect()

        # add factor
        xtrain = xtrain.reset_index()
        xtrain = create_field_location(xtrain)
        xtrain['Field_Location'] = xtrain['Field_Location'].astype('category')
        xtrain = xtrain.set_index(['Env', 'Hybrid'])
        xval = xval.reset_index()
        xval = create_field_location(xval)
        xval['Field_Location'] = xval['Field_Location'].astype('category')
        xval = xval.set_index(['Env', 'Hybrid'])
        xtest = xtest.reset_index()
        xtest = create_field_location(xtest)
        xtest['Field_Location'] = xtest['Field_Location'].astype('category')
        xtest = xtest.set_index(['Env', 'Hybrid'])
        
        print('# Features:', xtrain.shape[1])

    model = lgbm.LGBMRegressor(
        random_state=args.seed,
        max_depth=3
    )
    model.fit(xtrain, ytrain)
    

    # predict
    ypred_train = model.predict(xtrain)
    ypred = model.predict(xval)

    df_sub['Yield_Mg_ha'] = model.predict(xtest)
    df_sub.to_csv(f'output/2016/result/submission_{ma}.csv', index=False)

    pearson_train, _ = pearsonr(ytrain, ypred_train)
    pearson_val, _ = pearsonr(yval, ypred)
    rmse_train = np.sqrt(mean_squared_error(ytrain, ypred_train))
    rmse_val = np.sqrt(mean_squared_error(yval, ypred))
    print(f"Pearson correlation (train): {pearson_train:.4f}")
    print(f"Pearson correlation (val): {pearson_val:.4f}")
    print(f"RMSE (train): {rmse_train:.4f}")
    print(f"RMSE (val): {rmse_val:.4f}")
  

