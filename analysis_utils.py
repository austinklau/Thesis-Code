### UTILS FOR DATA ANALYSIS

# Imports
import numpy as np
import pandas as pd
import pickle
import cvxpy as cvx
from scipy import sparse


def rf_imp_aggregator(year_start, year_end, lag_labels, labels_extra, pathname, version): # years are inclusive
    '''Aggregates RF variable importances'''
    rf_imps = []
    rf_imps_extra = []
    years = [y for y in range(year_start, year_end+1)]
    for year in range(year_start, year_end+1):
        ftds = f'{year}-01-01'
        data_storage, _, _ = pickle.load(open(f'{pathname}/{ftds}_v{version}.p', 'rb'))

        rf_imp, _ = data_storage['rf_base_imp_tuple']
        rf_imps.append(rf_imp[lag_labels])

        rf_imp_extra, _ = data_storage['rf_extra_imp_tuple']
        rf_imps_extra.append(rf_imp_extra[labels_extra])

    combined = pd.concat(rf_imps, axis=1) # aggregate
    combined.columns = years
    combined_extra = pd.concat(rf_imps_extra, axis=1) # aggregate
    combined_extra.columns = years
    return combined, combined_extra


def extract_pred_accuracies(year_start, year_end, pathname, version):
    '''
    Extract prediction accuracies from saved data
    Returns annual averages and 
    '''
    years = [pd.to_datetime(y, format = '%Y') for y in range(year_start, year_end+1)]
    k_keys = ['K=5', 'K=10', 'K=50', 'K=100']
    accuracies_annual = [] # averages
    accuracies_extra_annual = [] # averages
    acc_ts = [] # time series
    acc_extra_ts = [] # time series
    for year in range(year_start, year_end+1):
        ftds = ftds = f'{year}-01-01'
        _, base_results_storage, extra_results_storage = pickle.load(open(f'{pathname}/{ftds}_v{version}.p', 'rb'))

        # Normal preds
        pred_accs = {} # average
        pred_ts = {} # time series
        for k in k_keys:
            acc = base_results_storage[k]['pred_accuracy']
            pred_accs.update({k:acc.mean()})
            pred_ts.update({k:acc})
        accuracies_annual.append(pred_accs)
        acc_ts.append(pd.DataFrame.from_dict(pred_ts)) # add df

        # Extra parameter preds
        pred_accs_extra = {} # average
        pred_extra_ts = {} # time series
        for k in k_keys:
            acc_extra = extra_results_storage[k]['pred_accuracy']
            pred_accs_extra.update({k:acc_extra.mean()})
            pred_extra_ts.update({k:acc_extra})
        accuracies_extra_annual.append(pred_accs_extra)
        acc_extra_ts.append(pd.DataFrame.from_dict(pred_extra_ts)) # add df
    
    # create dataframes and index by date
    accs_df = pd.DataFrame.from_dict(accuracies_annual)
    accs_extra_df = pd.DataFrame.from_dict(accuracies_extra_annual)
    accs_df.index = years 
    accs_extra_df.index = years

    # create time series dataframes
    return accs_df, accs_extra_df, acc_ts, acc_extra_ts


def extract_returns(year_start, year_end, pathname, version):
    '''Extract returns from saved data'''
    k_keys = ['K=5', 'K=10', 'K=50', 'K=100']
    # lists to be converted to dataframes
    raw_rets_list = []
    t_cost_rets_list = []
    raw_rets_extra_list = []
    t_cost_rets_extra_list = []
    for year in range(year_start, year_end+1):
        
        ftds = ftds = f'{year}-01-01'
        _, base_results_storage, extra_results_storage = pickle.load(open(f'{pathname}/{ftds}_v{version}.p', 'rb'))

        # Normal prets
        raw_rets = {}
        t_cost_rets = {}
        # Extra features rets
        raw_rets_extra = {}
        t_cost_rets_extra = {}

        for k in k_keys:
            # normal rets
            raw_rets.update({k:base_results_storage[k]['raw_rets']})
            t_cost_rets.update({k:base_results_storage[k]['t_cost_rets']})
            # extra feature rets
            raw_rets_extra.update({k:extra_results_storage[k]['raw_rets']})
            t_cost_rets_extra.update({k:extra_results_storage[k]['t_cost_rets']})
        
        # Appends
        last_date = pd.to_datetime(f'{year}-12-31') # last date of the year
        raw_rets_list.append(pd.DataFrame.from_dict(raw_rets).loc[:last_date])
        # print(raw_rets_list)
        t_cost_rets_list.append(pd.DataFrame.from_dict(t_cost_rets).loc[:last_date])
        raw_rets_extra_list.append(pd.DataFrame.from_dict(raw_rets_extra).loc[:last_date])
        t_cost_rets_extra_list.append(pd.DataFrame.from_dict(t_cost_rets_extra).loc[:last_date])

   
    # create dataframes and index by date
    raw_rets_df = pd.concat(raw_rets_list, axis=0)
    t_cost_rets_df = pd.concat(t_cost_rets_list, axis=0)
    raw_rets_extra_df = pd.concat(raw_rets_extra_list, axis=0)
    t_cost_rets_extra_df = pd.concat(t_cost_rets_extra_list, axis=0)
    return raw_rets_df, t_cost_rets_df, raw_rets_extra_df, t_cost_rets_extra_df


def sharpe_ratios_from_df(rets_df, start_date, end_date):
    '''Sharp ratios (using arithmetic returns) for a time series'''

    # rfree (fama-french dataset)
    ff_df = pd.read_csv('Data/fama_french.csv').set_index('date')
    ff_df.index = pd.to_datetime(ff_df.index)
    rf_df = ff_df['rf']

    # subtract out riskfree and calculate sharpe ratio
    df_minus_rfree = rets_df.subtract(rf_df[rets_df.index], axis=0).loc[start_date:end_date]
    sharpes = df_minus_rfree.mean().div(df_minus_rfree.std()) * np.sqrt(252) # annualize sharpe ratio
    return sharpes

def get_period_results(df, p_key, p_val, k_key='K=10'):
    '''Get annualized period results'''
    res_list = {}
    # p= SUB_PERIODS['09/08-12/09']
    df_cut = df[[k_key]].loc[p_val[0]:p_val[1]]
    num_days = len(df_cut)
    # median = df_cut.median() * 252
    # res_list.update({'Median': median})
    mean = (1 + df_cut.mean()) ** 252 -1
    res_list.update({'Mean': mean})
    sd = df_cut.std() * np.sqrt(252)
    res_list.update({'Std Dev': sd})
    sharpe = sharpe_ratios_from_df(df_cut, p_val[0], p_val[1])
    res_list.update({'Sharpe': sharpe})
    cum_rets_cut = np.cumprod(df_cut+1)
    cum_max = np.maximum.accumulate(cum_rets_cut, axis=0)
    mdd = np.max((cum_max - cum_rets_cut)/cum_max, axis=0)
    res_list.update({'MDD': mdd})
    res_list = pd.DataFrame.from_dict(res_list)
    res_list.index = [p_key]
    return res_list



### EXPERIMENTAL CODE ###
def trend_filter_regimes(df, start_idx, end_idx, lambd=1):
    '''
    Applies trend filtering algorithm to identify regimes
    start_idx and end_idx are INCLUSIVE
    Trend filtering is not good for predictions, but useful for analysis in post
    '''
    df_cut = df[start_idx:end_idx+1]
    y = df_cut['total_ret'].values
    n = len(y)
    one_vec = np.ones((1, n))
    D = sparse.spdiags(np.vstack((one_vec, -2*one_vec, one_vec)), range(3), m=n-2, n=n).toarray()  # spdiags(data, diags_to_set, m, n)

    x = cvx.Variable(n)
    objective = cvx.Minimize(.5* cvx.sum_squares(y-x)+ lambd * cvx.norm(D @ x, 1))
    prob = cvx.Problem(objective)

    # solver = cp.CVXOPT
    solver = cvx.ECOS
    prob.solve(solver=solver) #prob.solve() #  , verbose=True
    # plt.plot(np.vstack((y, x.value.flatten())).T)

    diffs = np.diff(x.value, n=1)
    diffs[0] = diffs[1]
    regs = np.array([0 if i >= 0 else 1 for i in diffs])
    regs = np.append(regs[0], regs)
    regimes = pd.DataFrame(regs, index=df_cut.index)
    regimes.columns = ['Regime_ind']
    regimes['Class'] = regimes.apply(lambda row: 'Growth' if row['Regime_ind'] == 0 else 'Contraction', axis=1)
    regimes['Trend'] = x.value
    return regimes


