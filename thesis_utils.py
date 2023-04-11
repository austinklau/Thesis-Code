### UTILITY FUNCTIONS FOR THESIS 

# Imports
import numpy as np
import pandas as pd
import pickle
import datetime
import itertools
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor


# Data Utils

def get_stock_data(db, permnos_master, start_date, end_date, first_trade_date):
    '''
    Gets stock data for a given timespan and set of permnos
    db = database connection
    permnos_master = master set of merno
    first_trade_date = date used to select permnos active when trade period starts
    start_date, end_date = start and end of time series
    '''
    # start_date_pd = pd.Timestamp(start_date).date()

    # get permnos in index at trade period start date and that have full data run
    current_permnos = permnos_master[(permnos_master['start'] < first_trade_date) & (permnos_master['ending'] >= first_trade_date)]['permno']
    # stringify permnos
    current_permnos_string = ", ".join(map(str,current_permnos.to_list()))

    query_template = f"""select permno, date, prc, ret, vol, shrout, hsiccd, bid, ask
                        from crsp.dsf
                        where permno in ({current_permnos_string})
                        and date >= '{start_date}' and date < '{end_date}' """

    data = db.raw_sql(query_template, date_cols=['date'])

    # DELETE INCOMPLETE DATA AND UPDATE PERMNOS LIST
    train_data = data[data['date'] < pd.to_datetime(first_trade_date)]
    train_data_nans = train_data[train_data.isnull()]
    bad_permnos = train_data_nans['permno'].unique()
    good_permnos = np.setdiff1d(current_permnos.values, bad_permnos)


    return data, good_permnos


def permno_start_ends(data, permnos):
    ''' 
    Helper function to get start and end dates for each permno
    [(start, end)] format
    Constrained by data and not listing/delisting
    '''

    d = []
    for permno in permnos:
        unique_dates = data[data['permno']==permno]['date']
        start = unique_dates.iloc[0]
        end = unique_dates.iloc[-1]
        d.append((start, end)) 
    return d


def generate_tradability_matrix(data, permnos, maxlag):
    ''' 
    Generate a boolean dataframe if each permno is tradable on that date
    Uses data as constraint, not listing/delisting (same as above)
    '''

    arbitrary_past_date = pd.Timestamp('1900-01-01')
    start_ends = permno_start_ends(data, permnos)
    dates = data['date'].unique()
    results = []
    for idx, date in enumerate(dates): # use each date as a response date
        target_date = date
        last_lag_date = dates[idx - maxlag - 1] if idx - maxlag - 1 >= 0 else arbitrary_past_date
        tradable = [((target_date <= x[1]) and (last_lag_date >= x[0])) for x in start_ends]
        results.append(tradable)
    results = pd.DataFrame(data=results, columns=permnos)
    results.set_index(dates, inplace=True)
    return results

def generate_total_returns_matrix(data):
    '''pivots raw data to get total returns matrix'''

    trunc = data[['date', 'permno','ret']]
    rets = trunc.pivot(index='date', columns='permno')
    rets.columns = rets.columns.droplevel()
    rets.index.name = None
    rets.columns.name = None

    # handle error codes by setting to zero
    rets.fillna(0, inplace=True)
    rets[rets < -1] = 0
    total_rets = (rets+1).cumprod(axis=0)
    total_rets.columns = total_rets.columns.astype(int)

    return total_rets

def generate_lag_features_targets_v2(data, permnos, lags, extra_data = None):
    ''' 
    Generate training lags + target response for a set of permnos for a given target date
    For every date, permno combination, there is a tuple of (predictors, targets_raw, targets_bool, tradability_matrix)
    '''
    # Map of PERMNO:industry (dummified to one-hot)
    permno_ind = data[['permno', 'ind']].drop_duplicates(subset='permno')
    ind_dummies = pd.get_dummies(permno_ind, prefix='IND', columns=['ind']).set_index('permno')
    # sic_feats = sic_dummies.columns.to_list()

    tradability_matrix = generate_tradability_matrix(data, permnos, lags[-1]) # get tradability matrix
    total_rets = generate_total_returns_matrix(data) # total returns matrix

    dates = data['date'].unique()


    predictors = {}
    predictors_extra = {}
    targets_raw = {}
    for permno in permnos:
        df_permno = total_rets[permno] # df for this permno
        pred_obs = [None] * len(dates)
        pred_obs_extra = [None] * len(dates)
        target_obs = [None] * len(dates)
        for idx, date in enumerate(dates):
            if tradability_matrix[permno].loc[date]:
                i = df_permno.index.get_loc(date) # feature index
                
                # Generate features and targets (changed to work using total returns)
                features = (df_permno.iloc[i-1]/df_permno.iloc[i-lags-1] - 1).values
                raw_target = df_permno.iloc[i]/df_permno.iloc[i-1] -1
                 
                pred_obs[idx] = features
                target_obs[idx] = raw_target

                # If including extra features
                if extra_data is not None:
                    j = extra_data.index.get_indexer([date],method='ffill')[0]
                    features_extra = np.append(features, extra_data.iloc[j-1].values) # get day before
                    features_extra = np.append(features_extra, ind_dummies.loc[permno].values) # append SIC code
              
                    pred_obs_extra[idx] = features_extra


        predictors.update({permno:pred_obs})
        targets_raw.update({permno:target_obs})
        # Extra feature
        if extra_data is not None:
            predictors_extra.update({permno:pred_obs_extra})

    # generate dataframes
    predictors = pd.DataFrame.from_dict(predictors, orient='columns')
    predictors.set_index(dates, inplace=True)
    targets_raw = pd.DataFrame.from_dict(targets_raw, orient='columns')
    targets_raw.set_index(dates, inplace=True)
    # Extra feature
    if extra_data is not None:
        predictors_extra = pd.DataFrame.from_dict(predictors_extra, orient='columns')
        predictors_extra.set_index(dates, inplace=True)
    

    # generate bools
    zero_series = pd.Series({permno: 0 for permno in permnos})
    targets_bool = targets_raw.apply(lambda row: row >= row.median() if sum(pd.notnull(row)) != 0 else zero_series, axis=1) * 1

    # null out non-tradables
    predictors = predictors.where(tradability_matrix.values)
    targets_raw = targets_raw.where(tradability_matrix.values)
    targets_bool = targets_bool.where(tradability_matrix.values)
    # Extra feature
    if extra_data is not None:
        predictors_extra = predictors_extra.where(tradability_matrix.values)

    # Conditional returns
    if extra_data is not None:
        return predictors, predictors_extra, targets_raw, targets_bool, tradability_matrix
    else:
        return predictors, None, targets_raw, targets_bool, tradability_matrix
    

def generate_training_matrices(features, targets, start_idx, end_idx, labels):
    ''' 
    Generate feature dataframe and target vector from dataframes
    [start_idx, end_idx] are indices of dataframes to select 
    '''

    feats_no_nulls = features.iloc[start_idx:end_idx+1].apply(lambda x: x[~pd.isna(x)].to_numpy(), axis=1).values
    feats_flat = np.vstack(np.concatenate(feats_no_nulls, axis=0))
    feats_df = pd.DataFrame(data=feats_flat, columns = labels)

    targs_no_nulls = targets.iloc[start_idx:end_idx+1].apply(lambda x: x[~pd.isna(x)].to_numpy(), axis=1).values
    targs_flat = np.concatenate(targs_no_nulls, axis=0).astype(int)

    return feats_df, targs_flat


# Trading Utils

def generate_features_for_one_day(features, day_idx, lag_labels):
    '''Generate feature dataframe + corrresponding indices for a single day of trading'''
    
    day_features = features.iloc[day_idx]
    day_features = day_features[~pd.isna(day_features)]
    day_features_permnos = day_features.index.values

    feats_flat = np.vstack(day_features.values)
    feats_df = pd.DataFrame(data=feats_flat, columns = lag_labels)

    return feats_df, day_features_permnos


def generate_positions(features, start_idx, end_idx, lag_labels, model, k):
    ''' 
    Generate top and flop arrays for daily positions using features array
    Since features are lagged by 1, positions start at this date index
    k = # tops/flops to use
    '''
   
    # Top and Flop lists
    tops_list = []
    flops_list = []

    for idx in range(start_idx, end_idx+1):
        # get features
        feats_temp, permnos_temp = generate_features_for_one_day(features, idx, lag_labels)
        # predict model
        preds_temp = model.predict(feats_temp)

        # get top and flop predicitons
        top_k_idx = np.sort(np.argpartition(preds_temp, -k)[-k:])
        flop_k_idx = np.sort(np.argpartition(preds_temp, k)[:k])
        
        # Generate long and short positions
        top_k = preds_temp[top_k_idx]
        top_k_permnos = permnos_temp[top_k_idx]
        tops_list.append(dict(zip(top_k_permnos, top_k))) 

        flop_k = preds_temp[flop_k_idx]
        flop_k_permnos = permnos_temp[flop_k_idx]
        flops_list.append(dict(zip(flop_k_permnos, flop_k))) 

    # top and flop dataframes (not same cols as the features)
    tops_raw = pd.DataFrame.from_dict(tops_list)
    flops_raw = pd.DataFrame.from_dict(flops_list)

    # positions to +/-1 & setting cols to match features
    col_names = features.iloc[:0,:].copy()
    
    tops = pd.concat([col_names, tops_raw]).notnull() * 1
    flops = pd.concat([col_names, flops_raw]).notnull() * -1

    # index to actual holding days 
    date_index = features.index[start_idx:end_idx+1]
    tops.set_index(date_index, inplace=True)
    flops.set_index(date_index, inplace=True)
        
    return tops, flops
 

def count_transactions(position_matrix, k):
    '''
    Counts transactions per day for a given matrix (opens + closes)
    Input =  (tops or flops)
    '''
    open_diffs = position_matrix.abs().diff(periods=1,axis=0)
    open_diffs[open_diffs == -1]  = 0 # null out closes
    opens = open_diffs.sum(axis=1)
    opens[0] = k # fill first

    close_diffs = position_matrix.abs().diff(periods=-1, axis=0)
    close_diffs[close_diffs == -1] = 0 # null out opens
    closes = close_diffs.sum(axis=1)
    closes[-1] = k # fill last

    transactions = opens + closes
    return transactions


def simulate_returns(tops, flops, k, returns, long_t_cost = 0, short_t_cost = 0):
    '''
    Simulates trades for a period
    Trade period is defined by length of tops and flops matrices (trading starts on first day). Returns truncated to match
    k = # of tops/flops
    '''
    # count daily transactions for tops and flops
    tops_transactions = count_transactions(tops, k)
    flops_transactions = count_transactions(flops, k)

    # truncate returns to match position booleans
    start_date = tops.iloc[0].name
    end_date = tops.iloc[-1].name
    returns = returns.loc[start_date:end_date]

    # calculate daily returns
    raw_rets = (returns * tops).sum(axis=1) + (returns * flops).sum(axis=1)
    raw_rets /= (2 * k) # normalize by position size

    # Currently: t-costs per unit of trade
    t_cost_rets = raw_rets - (tops_transactions * long_t_cost + flops_transactions * short_t_cost ) / (2*k) # factor in t-costs
    
    return raw_rets, t_cost_rets


def simulate_trade_period(features, targets_bool, returns, trade_start_idx, trade_end_idx, labels, rf, k, t_cost_long=0, t_cost_short=0):
    '''Gets return series for a trading period'''
    # get tops and flops
    tops, flops = generate_positions(features, trade_start_idx, trade_end_idx, labels, rf, k)

    # prediction accuracy
    pred_accuracy = evaluate_prediction_accuracy(tops, flops, targets_bool, k)

    # returns
    raw_rets, t_cost_rets = simulate_returns(tops, flops, k, returns, t_cost_long, t_cost_short)
    
    return raw_rets, t_cost_rets, pred_accuracy, tops, flops


def evaluate_prediction_accuracy(tops, flops, targets_bool, k):
    '''Gets series of prediciton accuracy over time'''
    targets_bool_inverse = targets_bool -1
    top_correct = (tops * targets_bool.loc[tops.index]).sum(axis=1)
    # top_correct_sum = top_correct.mean()/k
    flop_correct = (flops * targets_bool_inverse.loc[flops.index]).sum(axis=1)
    # flop_correct_sum = flop_correct.mean()/k
    pct_correct = (top_correct + flop_correct) / (2 * k)
    return pct_correct


def generate_vix_features(vix_df, trade_start_date, qs = [0.25, 0.5, 0.75]): # inclusive, starts from inception
    '''Classifies VIX based ond data up to end of training period (NOT USED)'''
    trade_start_idx = vix_df.index.get_loc(trade_start_date)
  
    vix_known = vix_df[:trade_start_idx] # known vix
    quartiles = np.quantile(vix_known['CLOSE'], q = qs)
    vix_df.loc[:,'Regime'] = vix_df.apply(lambda row: 'Low' if row['CLOSE'] < quartiles[0] else ('High' if row['CLOSE'] > quartiles[2] else 'Normal'), axis=1)

    # One hot encode
    one_hot = pd.get_dummies(vix_df['Regime'], prefix = 'Vol')
    vix_one_hot = pd.concat([vix_df, one_hot], axis=1)

    return quartiles, vix_one_hot


def tree_importances(rf, labels):
    '''Tree importance for random forest'''
    importances = rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
    # individual_importances = [tree.feature_importances_ for tree in rf.estimators_]
    forest_importances = pd.Series(importances, index=labels).sort_values(ascending=False) 
    return (forest_importances, std)


def run_simulation_factors(first_trade_date_string, ks, num_trees, max_features, max_depth, lags, num_train_days, num_test_days, t_cost_long=0, t_cost_short = 0):
    '''Runs simulation but includes fama-french factors and FAMA industry codes'''

    # Load permnos and data
    data, current_permnos = pickle.load(open(f"Data/data_for{first_trade_date_string}_v2.p", "rb"))

    # extra factors to use in training
    extra_factors_list = ['VIX', 'MKT', 'SMB', 'HML', 'RMW', 'CMA']

    # vix df
    vix_df = pd.read_csv('Data/VIX_History.csv').set_index('DATE')
    vix_df.index.rename('date', inplace=True)
    vix_df.index = pd.to_datetime(vix_df.index)

    # ff df
    ff_df = pd.read_csv('Data/fama_french.csv').set_index('date')
    ff_df.index = pd.to_datetime(ff_df.index)

    # combine vix + ff
    extra_factors_df = ff_df.merge(vix_df['CLOSE'], how='inner', left_on = ff_df.index, right_on = vix_df.index).set_index('key_0')
    extra_factors_df.rename(columns={'CLOSE':'VIX'}, inplace=True)
    extra_factors_df.index.rename('Date', inplace=True)
    extra_factors = extra_factors_df[extra_factors_list]

    # Industry list
    permno_ind = data[['permno', 'ind']].drop_duplicates(subset='permno')
    ind_dummies = pd.get_dummies(permno_ind, prefix='IND', columns=['ind']).set_index('permno')
    ind_names_list = ind_dummies.columns.to_list()

    lag_labels = [f"R({i})" for i in lags] # Get lag labelsnum preds, based on period lag
    labels_extra = lag_labels + extra_factors_list + ind_names_list
    # labels_extra = lag_labels + ['Vol_Low','Vol_Normal', 'Vol_High']


    # Generate features (and extra features)
    features, features_extra, returns, targets_bool, tradability = generate_lag_features_targets_v2(data, current_permnos, lags, extra_factors)


    # Get endpoints of training period FROM FEATURES MATRIX
    trade_start_idx = features.index.get_indexer([pd.Timestamp(first_trade_date_string)],method='backfill')[0]
    train_end_idx = trade_start_idx-2
    train_start_idx = train_end_idx-num_train_days + 1

    # Last date of year
    # if first_trade_date_string == '2022-01-01':
    trade_end_idx = features.index.get_indexer([pd.Timestamp(f'{first_trade_date_string[:4]}-12-31')], method='ffill')[0] # edge case

    # Training data
    feats_train, targs_train = generate_training_matrices(features, targets_bool, train_start_idx, train_end_idx, lag_labels) # Data for training BASE TREE
    feats_extra_train, _ = generate_training_matrices(features_extra, targets_bool, train_start_idx, train_end_idx, labels_extra) # Data for training TREE WITH EXTRA FEATS

    # STORAGE
    data_storage = {}
    data_storage.update({'data':data})
    data_storage.update({'current_permnos':current_permnos})
    data_storage.update({'features': features})
    data_storage.update({'features_extra': features_extra})
    data_storage.update({'returns': returns})
    data_storage.update({'targets_bool': targets_bool})
    data_storage.update({'tradability': tradability})
    data_storage.update({'trade_start_idx': trade_start_idx})
    data_storage.update({'trade_end_idx': trade_end_idx})
    data_storage.update({'train_start_idx': train_start_idx})

    # Base Model Training
    rf_base = RandomForestRegressor(max_features=max_features, max_depth=max_depth, n_estimators = num_trees, random_state=69, n_jobs=-1) # , n_jobs=-1 
    rf_base.fit(feats_train, targs_train)
    rf_base_imp_tuple = tree_importances(rf_base, lag_labels)

    # Extra Features Model Training
    rf_extra = RandomForestRegressor(max_features=max_features, max_depth=max_depth, n_estimators = num_trees, random_state=69, n_jobs=-1)
    rf_extra.fit(feats_extra_train, targs_train)
    rf_extra_imp_tuple = tree_importances(rf_extra, labels_extra)

    # STORAGE
    tree_storage = {}
    tree_storage.update({'rf_base':rf_base})
    tree_storage.update({'rf_extra':rf_extra})

    # more data storage
    data_storage.update({'rf_base_imp_tuple':rf_base_imp_tuple})
    data_storage.update({'rf_extra_imp_tuple':rf_extra_imp_tuple})

    # Simulation
    base_results_storage = {}
    extra_results_storage = {}
    for k in ks:
        # Regular tree
        raw_rets, t_cost_rets, pred_accuracy, tops, flops = simulate_trade_period(features, targets_bool, returns, trade_start_idx, trade_end_idx, lag_labels, rf_base, k, t_cost_long, t_cost_short)
        temp_dict = {'raw_rets': raw_rets, 't_cost_rets': t_cost_rets, 'pred_accuracy': pred_accuracy, 'tops':tops, 'flops':flops}
        base_results_storage.update({f'K={k}':temp_dict})

        # Extra tree
        raw_rets2, t_cost_rets2, pred_accuracy2, tops2, flops2 = simulate_trade_period(features_extra, targets_bool, returns, trade_start_idx, trade_end_idx, labels_extra, rf_extra, k, t_cost_long, t_cost_short)
        temp_dict2 = {'raw_rets': raw_rets2, 't_cost_rets': t_cost_rets2, 'pred_accuracy': pred_accuracy2,'tops':tops2, 'flops':flops2}
        extra_results_storage.update({f'K={k}':temp_dict2})
    
    return data_storage, base_results_storage, extra_results_storage, tree_storage