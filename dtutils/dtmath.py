# dtmath.py
# Statistics, transformations, filters and other math

import pandas as pd
import numpy as np
import pickle
import pdb
from scipy.linalg import cholesky, solve_triangular
from sklearn.covariance import MinCovDet
from typing import Dict, Callable, Tuple

##########################################################
########## Covariance Matrix Estimation Methods ##########
##########################################################

# 1. Default estimation - all values
def est_cov_std(df: pd.DataFrame):
    mean_vec = df.mean()
    centered = df - mean_vec
    cov = np.cov(centered.T)
    return mean_vec, cov, len(df)

# 2. Minimum determinant estimation
def est_cov_mcd(df: pd.DataFrame, support: float = 0.8):
    mcd = MinCovDet(support_fraction=support).fit(df)
    cov = mcd.covariance_
    mean_vec = pd.Series(mcd.location_, index=df.columns)
    return mean_vec, cov, len(df)

# 3. Estimatioin by distance from zero (by modulus)
def est_cov_dist(df: pd.DataFrame, quantile: float = 0.8):
    norms = np.linalg.norm(df.values, axis=1)
    threshold = np.quantile(norms, 1 - quantile)
    selected = df[norms >= threshold]
    mean_vec = selected.mean()
    cov = np.cov((selected - mean_vec).T)
    return mean_vec, cov, len(df)

# Wrapper for methods
def est_cov(df: pd.DataFrame, method='std', **kwargs):

    """
    A unified wrapper for covariance estimation.
    
    Arguments:
    - df : DataFrame with measurements
    - method : {'std', 'mcd', 'dist'}
    - kwargs : method-specific arguments

    Returns:
    - vector of mean values
    - covariance matrix
    - number of measurements in the sample
    """
    
    methods: dict[str, Callable] = {
        'std': est_cov_std,
        'mcd': est_cov_mcd,
        'dist': est_cov_dist,
    }
    
    if method not in methods:
        raise ValueError(f"Unknown evaluation method: {method}")
    
    return methods[method](df, **kwargs)

def est_cov_by_group(df: pd.DataFrame, group_map: pd.Series, method='std', **kwargs):

    """
    Estimation of the covariance matrix and means across time clusters.

    Arguments:
    - df: DataFrame with measurements (rows are points in time, columns are variables, in our case sensors)
    - group_map: Series, where each time is associated with a cluster
                 df and group_map must have a compatible index (DatetimeIndex)
    - method: {'std', 'mcd', 'dist'}
    - kwargs: parameters for the method

    Returns:
    - dictionary of cluster means
    - dictionary of covariance matrices by clusters
    - dictionary of sample sizes by cluster
    """

    mean_by_group = {}
    cov_by_group = {}
    n_by_group = {}

    for g in sorted(group_map.dropna().unique()):
        idx = group_map[group_map == g].index
        df_group = df.loc[idx].dropna()
        if df_group.empty:
            continue
        try:
            mean_vec, cov_mat, n = est_cov(df_group, method=method, **kwargs)
            mean_by_group[g] = mean_vec
            cov_by_group[g] = cov_mat
            n_by_group[g] = n
        except Exception as e:
            print(f"Ошибка в группе {g}: {e}")
            continue

    return mean_by_group, cov_by_group, n_by_group

def save_cov(filename, group_mean, group_cov, group_n):

    """
    Saves dictionaries with evaluations into one file.

    Arguments:
    - filename: path to the file (for example, 'covariances.pkl')
    - group_mean: dictionary of means
    - group_cov: dictionary of covariance matrices
    - group_n: dictionary of sample sizes
    """

    data = {
        'mean': group_mean,
        'cov': group_cov,
        'n': group_n
    }
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_cov(filename):

    """
    Loads dictionaries with evaluations from a file.

    Arguments:
    - filename: path to file

    Returns:
    - group_mean, group_cov, group_n
    """

    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data['mean'], data['cov'], data['n']



def whiten_by_group(df: pd.DataFrame, group_map: pd.Series, 
                    group_mean: dict, group_cov: dict):

    """
    Z-transformation by time groups.

    Arguments:
    - df: DataFrame with measurements
    - group_map: Series, where each time is associated with a group number
    - group_mean: dictionary of means ​​by group
    - group_cov: dictionary of covariance matrices by group

    Returns:
    - DataFrame with Z-transformed values
    """

    z_all = []

    for g in sorted(group_map.dropna().unique()):
        idx = group_map[group_map == g].index
        df_group = df.loc[idx].dropna()
        if df_group.empty or g not in group_mean or g not in group_cov:
            continue
        mean_vec = group_mean[g]
        cov_mat  = group_cov[g]

        try:
            L = cholesky(cov_mat, lower=True)
        except np.linalg.LinAlgError:
            print(f"The covariance matrix is ​​not positive definite for the group {g}")
            continue

        L_inv = solve_triangular(L, np.eye(L.shape[0]), lower=True)
        centered = df_group - mean_vec
        z_values = centered.values @ L_inv.T

        z_df = pd.DataFrame(z_values, index=df_group.index,
                            columns=[f'z_{col}' for col in df_group.columns])
        z_all.append(z_df)

    if z_all:
        return pd.concat(z_all).sort_index()
    else:
        return pd.DataFrame()

##############################
########## Filters ###########
##############################

def alarm_filter(series, crit_on, crit_off, window_size=12):

    """
    Hysteresis filter for signals based on smoothed z-squared value.
    Returns Series with 1/0 — where the alarm is active.
    """

    moving_avg = series.rolling(window=window_size, min_periods=1).mean()
    
    alarm_state = 0
    alarm_flags = []

    for val in moving_avg:
        if not alarm_state and val > crit_on:
            alarm_state = 1
        elif alarm_state and val < crit_off:
            alarm_state = 0
        alarm_flags.append(alarm_state)

    return pd.Series(alarm_flags, index=series.index)

######################################
########## Transformations ###########
######################################

def wht(zval, df):
    # Wilson-Hilferty transformation
    return (zval/df)**(1/3)


def f_stat(z2: pd.Series, eval_map: pd.Series, nobs, df: int) -> pd.Series:

    """
    Rescales Z^2 values to F-distribution using formula:
        F = ((n - df) / (df * (n - 1))) * Z^2
    where:
        z2       — timestamp-indexed Z^2 values
        eval_map — pd.Series: timestamp -> time cluster
        nobs     — dict or Series: number of observations for each time cluster
        df       — degrees of freedom.
    Returns Series with transformed values.
    """

    # Connecting data in DataFrame
    df_combined = pd.DataFrame({
        'z2': z2,
        'cluster': eval_map
    })

    # Obtaining the corresponding n for each time index
    df_combined['n'] = df_combined['cluster'].map(nobs)

    # Calculating the coefficient and the final value
    numerator = df_combined['n'] - df
    denominator = df * (df_combined['n'] - 1)
    df_combined['f_stat'] = (numerator / denominator) * df_combined['z2']

    return df_combined['f_stat']