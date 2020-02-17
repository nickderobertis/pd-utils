import pandas as pd
import numpy as np


def _to_list_if_str(var):
    if isinstance(var, str):
        return [var]
    else:
        return var

def _to_series_if_str(df, i):
    if isinstance(i, pd.Series):
        s = i
    elif isinstance(i, str):
        s = df[i]
    else:
        raise ValueError('Please provide a str, list of strs, or a list of pd.Series for byvars')
    return s

def _to_name_if_series(i):
    if isinstance(i, pd.Series):
        return i.name
    else:
        return i


def split(df, keepvars, keyvar='__key_var__'):
    """
    Splits a dataframe into a list of arrays based on a key variable
    """
    small_df = df[[keyvar] + keepvars]
    arr = small_df.values
    splits = []
    for i in range(arr.shape[0]):
        if i == 0: continue
        if arr[i,0] != arr[i-1,0]: #different key
            splits.append(i)
    return np.split(arr[:,1:], splits)
