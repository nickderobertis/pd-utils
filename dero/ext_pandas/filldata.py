import pandas as pd
import numpy as np
from functools import partial
from itertools import product

from .pdutils import _to_list_if_str

def fillna_by_groups_and_keep_one_per_group(df, byvars, exclude_cols=None, str_vars='first', num_vars='mean'):
    """
    WARNING: do not use if index is important, it will be dropped
    """
    byvars = _to_list_if_str(byvars)
    if exclude_cols:
        exclude_cols = _to_list_if_str(exclude_cols)

    df = fillna_by_groups(df, byvars, exclude_cols=exclude_cols, str_vars=str_vars, num_vars=num_vars)
    _drop_duplicates(df, byvars)

    return df


def fillna_by_groups(df, byvars, exclude_cols=None, str_vars='first', num_vars='mean'):
    """
    WARNING: do not use if index is important, it will be dropped
    """
    byvars = _to_list_if_str(byvars)

    if exclude_cols:
        cols_to_fill = [col for col in df.columns if (col not in exclude_cols) and (col not in byvars)]
        concat_vars = byvars + exclude_cols
    else:
        cols_to_fill = [col for col in df.columns if col not in byvars]
        concat_vars = byvars

    _fill_data = partial(_fill_data_for_series, str_vars=str_vars, num_vars=num_vars)

    out_dfs = []
    for group, group_df in df[byvars + cols_to_fill].groupby(byvars, as_index=False):
        out_dfs.append(group_df.apply(_fill_data, axis=1))

    filled = pd.concat(out_dfs, axis=0).reset_index(drop=True)

    filled = _restore_nans_after_fill(filled) #_fill_data places -999.999 in place of nans, now convert back


    return filled

def add_missing_group_rows(df, fill_id_cols):
    fill_ids = [df[fill_id_col].unique() for fill_id_col in fill_id_cols]
    index_df = pd.DataFrame([i for i in product(*fill_ids)], columns=fill_id_cols)

    return index_df.merge(df, how='left', on=fill_id_cols)

def drop_missing_group_rows(df, fill_id_cols):
    drop_subset = [col for col in df.columns if col not in fill_id_cols]
    return df.dropna(subset=drop_subset, how='all')

def _fill_data_for_series(series, str_vars='first', num_vars='mean'):
    index = _get_non_nan_value_index(series, str_vars)
    if index is None:
        # All nans, can't do anything but return back nothing
        # But transform ignores nans in the output and then complains when the sizes don't match.
        # So instead, put a placeholder of -999.999
        return pd.Series([-999.999 for i in range(len(series))])
    # handle numeric
    if series.dtype in (np.float64, np.int64):
        if num_vars in ('first', 'last'):
            # Overwrite index for that of num vars if not using the same value as for str vars
            if num_vars != str_vars:
                index = _get_non_nan_value_index(series, num_vars)
            return _fill_data_for_str_series(series, non_nan_index=index)
        return _fill_data_for_numeric_series(series, fill_function=num_vars)
    # handle strs
    else:
        return _fill_data_for_str_series(series, non_nan_index=index)


def _fill_data_for_numeric_series(series, fill_function='mean'):
    return series.fillna(series.agg(fill_function))


def _fill_data_for_str_series(series, non_nan_index):
    fill_value = series.loc[non_nan_index]

    return series.fillna(fill_value)

def _get_non_nan_value_index(series, first_or_last):
    if first_or_last == 'first':
        return series.first_valid_index()
    elif first_or_last == 'last':
        return series.last_valid_index()
    else:
        raise ValueError("Did not pass 'first' or 'last'")


def _restore_nans_after_fill(df):
    """
    -999.999 was used as a missing representation as pandas can not handle nans in transform.
    Convert back to nan now
    """
    return df.applymap(lambda x: np.nan if x == -999.999 else x)

def _drop_duplicates(df, byvars):
    """
    Note: inplace
    """
    df.drop_duplicates(subset=byvars, inplace=True)