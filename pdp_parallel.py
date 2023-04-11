# Standard Imports
from IPython.display import display
import numpy as np
import pandas as pd
import pickle
import time
import datetime
import itertools
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib as mpl
from sklearn import metrics
from sklearn import linear_model as lm
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence
from joblib import Parallel, delayed
from mpl_toolkits.mplot3d import Axes3D

# Import Custom Utilities
import thesis_utils as utils
import analysis_utils as au

# Master Params
FIRST_TRADE_STRINGS = [f'{year}-01-01' for year in range(1993, 2023)]
KS = [5, 10, 50, 100]
NUM_TREES = 10
MAX_FEATURES = 6
MAX_DEPTH = 20
LAGS = np.concatenate((np.arange(1,21), (np.arange(2,13) * 20)))
TRAIN_DAYS = 504 
TRADE_DAYS = 252
T_COST = 0.001 # half-turn t-cost (need to scale returns if we are doing other analysis)

# Path for source files
VERSION = '3'
# DATA_PATHNAME = f'Adroit_results_v{VERSION}'
# RESULTS_PATHNAME = f'Analysis_v{VERSION}'


extra_factors_list = ['VIX', 'MKT', 'SMB', 'HML', 'RMW', 'CMA']
ind_names_list = ['IND_BusEq', 'IND_Chems', 'IND_Durbl', 'IND_Enrgy', 'IND_Fin', 'IND_Hlth', 'IND_Manuf', 'IND_NoDur', 'IND_Other', 'IND_Shops', 'IND_Telcm', 'IND_Utils']
lag_labels = [f"R({i})" for i in LAGS] # Get lag labelsnum preds, based on period lag
labels_extra = lag_labels + extra_factors_list + ind_names_list


# helper function to get PDPs
def PDP_retriever(i):
    ftds = FIRST_TRADE_STRINGS[i]
    dataname = f'Results/{ftds}_v{VERSION}.p' # fix
    treename = f'Results/tree_{ftds}_v{VERSION}.p' # fix 

    tree_storage = pickle.load(open(treename, 'rb'))
    data_storage, _,_ = pickle.load(open(dataname, 'rb'))


    lr = tree_storage['rf_base']
    lrf = tree_storage['rf_extra']
    features = data_storage['features']
    features_extra = data_storage['features_extra']
    targets_bool = data_storage['targets_bool']
    train_start_idx = data_storage['train_start_idx']
    trade_start_idx = data_storage['trade_start_idx']
    train_end_idx = trade_start_idx -1

    # feats_train, targs_train = utils.generate_training_matrices(features, targets_bool, train_start_idx, train_end_idx, lag_labels) 
    feats_extra_train, _ = utils.generate_training_matrices(features_extra, targets_bool, train_start_idx, train_end_idx, labels_extra) 

    longs = [20, 21, 22, 31]
    n = len(longs)

    # LRF model
    pdp = {}
    for i, name in enumerate(labels_extra[:-12]):
        results = partial_dependence(lrf, feats_extra_train, [i], grid_resolution=100)
        pdp.update({name:results})

    # LRF model 2d
    pdp_2d = {}
    # short lags 1-5
    for i in range(5):
        for j in range(i):
            results = partial_dependence(lrf, feats_extra_train, [j,i], kind='average', grid_resolution=10)
            name = f"{labels_extra[j]} - {labels_extra[i]}"
            pdp_2d.update({name:results})
    # long lags and VIX interact w/ short lags
    for i in range(5):
        for j in range(n):
            results = partial_dependence(lrf, feats_extra_train, [longs[j],i], kind='average', grid_resolution=10)
            name = f"{labels_extra[longs[j]]} - {labels_extra[i]}"
            pdp_2d.update({name:results})

    results_dict = {'1D':pdp, '2D':pdp_2d}
    # final_pdp = {ftds:results_dict}
    return results_dict
    # pickle.dump(final_pdp, open(f"Results/_tree_{ftds}_pdp_v{VERSION}.p", "wb" ))


# Run jobs in parallel
n_years = len(FIRST_TRADE_STRINGS)
pdps = Parallel(n_jobs=16)(delayed(PDP_retriever)(i) for i in range(n_years))
years = [f'{year}' for year in range(1993, 2023)]
pdps_dict = {y:p for y,p in zip(years, pdps)}
pickle.dump(pdps_dict, open(f"Results/_pdps_v{VERSION}.p", "wb" ))


