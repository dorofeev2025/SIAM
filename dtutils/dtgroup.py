# dtgroup.py
# Functions for grouping time intervals

import pandas as pd
import numpy as np
from typing import Union
from sklearn.cluster import KMeans
import pdb


def group_all(index: pd.DatetimeIndex) -> pd.Series:
    
    # All timestamps fall into one cluster (0).
    
    return pd.Series(0, index=index, name='cluster')


def group_by_hour(index: pd.DatetimeIndex) -> pd.Series:
    
    # Clustering by hour (0..23).
    
    return pd.Series(index.hour, index=index, name='cluster')


def group_by_n_hours(index: pd.DatetimeIndex, n: int = 4) -> pd.Series:
    """
    Clustering into groups of n hours.
    For example, when n=3:
      00:00–02:59 → 0, 03:00–05:59 → 1 etc.
    """
    clusters = (index.hour // n).astype(int)
    return pd.Series(clusters, index=index, name='cluster')


def group_by_pattern(index: pd.DatetimeIndex, pattern: Union[list, np.ndarray]) -> pd.Series:
    """
    Repeats the given clustering pattern throughout the sample.

    Arguments:
    - index: DatetimeIndex, to which the scheme will be applied
    - pattern: a list or array of clusters (for example, [0, 0, 1, 1, 2])

    Returns:
    - Series with a repeating cluster for each timestamp.

    Example of use:
    pattern = [0]*7 + [1]*3 + [2]*4 + [3]*4 + [4]*4 + [0]*2  # 24 elements
    group_series = group_by_pattern(df.index, pattern)
    """
    pattern = np.asarray(pattern)
    n = len(index)
    k = len(pattern)
    reps = int(np.ceil(n / k))
    tiled = np.tile(pattern, reps)[:n]  # repeat and trim
    return pd.Series(tiled, index=index, name='cluster')


def cluster_by_kmeans(df: pd.DataFrame, n_clusters: int = 6, random_state: int = 42) -> pd.Series:
    """
    Clustering of time moments by feature values ​​using KMeans.

    Arguments:
    - df: DataFrame (index — DatetimeIndex, rows are time points, columns are features/sensors)
    - n_clusters: number of clusters
    - random_state: parameter for reproducibility

    Returns:
    - Series with clusters
    """
    df_clean = df.dropna()
    if df_clean.empty:
        raise ValueError("No data for clustering")

    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = model.fit_predict(df_clean.values)

    return pd.Series(labels, index=df_clean.index, name='cluster')


def cluster_by_hour(df: pd.DataFrame, n_clusters: int = 6, random_state: int = 42) -> pd.Series:
    """
    Clustering by hour of day using sensor readings.

    Arguments:
    - df: DataFrame (index — DatetimeIndex, rows are time points, columns are features/sensors)
    - n_clusters: number of clusters
    - random_state: parameter for reproducibility

    Returns:
    - Series: displays every hour of the day (0–23) → cluster
    """
    df_clean = df.dropna()
    if df_clean.empty:
        raise ValueError("No data for clustering")

    # Grouping by hour of day
    df_clean = df_clean.copy()
    df_clean["hour"] = df_clean.index.hour
    grouped = df_clean.groupby("hour").mean()  # averaging across sensors for each hour of the day

    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = model.fit_predict(grouped.values)
    return pd.Series(labels, index=grouped.index, name="cluster")


def cluster_by_weektime(df: pd.DataFrame, n_clusters: int = 6, random_state: int = 42) -> pd.Series:
    """
    Clustering of time points by time within a week, 
    and the similarity is in sensory meanings.

    Arguments:
    - df: DataFrame with DatetimeIndex and features (sensors)
    - n_clusters: number of clusters
    - random_state: parameter for reproducibility

    Returns:
    - Series: cluster for each time point df.index
    """
    df_clean = df.dropna()
    if df_clean.empty:
        raise ValueError("No data for clustering")

    # Let's set aside time during the week
    weektime_index = df_clean.index.dayofweek * 24 + df_clean.index.hour  # 0..167
    df_clean = df_clean.copy()
    df_clean['weektime'] = weektime_index

    # Grouping by time within a week and averaging
    group_means = df_clean.groupby('weektime').mean()

    # Clustering of these average values
    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    cluster_labels = model.fit_predict(group_means.values)

    # Returning the hour of the week display → cluster
    return pd.Series(cluster_labels, index=group_means.index, name="cluster")