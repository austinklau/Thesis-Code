# Standard Imports
import numpy as np
import os
import pandas as pd
import pickle
import time
from joblib import Parallel, delayed

# Import Utilities
import thesis_utils as utils

# Master Parameters
FIRST_TRADE_STRINGS = [f'{year}-01-01' for year in range(1993, 2023)]
KS = [5, 10, 50, 100]
NUM_TREES = 1000
MAX_FEATURES = 6
MAX_DEPTH = 20
LAGS = np.concatenate((np.arange(1,21), (np.arange(2,13) * 20)))
TRAIN_DAYS = 504 
TRADE_DAYS = 252
T_COST = 0.001 # half-turn t-cost (10 bps)

# Run helper for parallelism
def run_helper(i):
    first_trade_date_string = FIRST_TRADE_STRINGS[i]
    start_time = time.time()
    print(f'Starting run for {first_trade_date_string}')
    # Run simulation
    data_storage, base_results_storage, extra_results_storage, tree_storage = \
        utils.run_simulation_factors(first_trade_date_string, KS, NUM_TREES, MAX_FEATURES, MAX_DEPTH, LAGS, TRAIN_DAYS, TRADE_DAYS, T_COST, T_COST)
    z = (data_storage, base_results_storage, extra_results_storage)
    pickle.dump(z, open(f"Results/{first_trade_date_string}_v3.p", "wb" ))
    pickle.dump(tree_storage, open(f"Results/tree_{first_trade_date_string}_v3.p", "wb" ))
    time_delta = round((time.time()- start_time) / 60,2)
    print(f'Finished {first_trade_date_string} in {time_delta} minutes')

# Run jobs in parallel
n_years = len(FIRST_TRADE_STRINGS)
Parallel(n_jobs=-1)(delayed(run_helper)(i) for i in range(n_years))