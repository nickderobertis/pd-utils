import pandas as pd
import numpy as np
from functools import partial

from .pdutils import _to_list_if_str

def fill_data_by_groups_and_drop_duplicates(df, byvars, exclude_cols=None, str_vars='first', num_vars='mean'):
    """

    """
    byvars = _to_list_if_str(byvars)
    if exclude_cols:
        exclude_cols = _to_list_if_str(exclude_cols)

    df = fill_data_by_groups(df, byvars, exclude_cols=exclude_cols, str_vars=str_vars, num_vars=num_vars)
    _drop_duplicates(df, byvars)

    return df


def fill_data_by_groups(df, byvars, exclude_cols=None, str_vars='first', num_vars='mean'):
    """

    """
    byvars = _to_list_if_str(byvars)

    if exclude_cols:
        cols_to_fill = [col for col in df.columns if col not in exclude_cols]
        concat_vars = byvars + exclude_cols
    else:
        cols_to_fill = [col for col in df.columns]
        concat_vars = byvars

    _fill_data = partial(_fill_data_for_series, str_vars=str_vars, num_vars=num_vars)

    filled = df[byvars + cols_to_fill].groupby(byvars).transform(_fill_data)

    # Filled is of the same dimensions as df but is missing byvars and exclude_cols. Add them back
    filled = pd.concat([df[concat_vars], filled], axis=1)

    return filled


def _fill_data_for_series(series, str_vars='first', num_vars='mean'):
    # All nans, can't do anything but return back nans
    if pd.isnull(series).all():
        return series
    # handle numeric
    if series.dtype in (np.float64, np.int64):
        return _fill_data_for_numeric_series(series, fill_function=num_vars)
    # handle strs
    else:
        return _fill_data_for_str_series(series, first_or_last=str_vars)


def _fill_data_for_numeric_series(series, fill_function='mean'):
    return series.fillna(series.agg(fill_function))


def _fill_data_for_str_series(series, first_or_last='first'):
    fill_value = _get_one_non_nan_from_series(series, first_or_last=first_or_last)

    return series.fillna(fill_value)


def _get_one_non_nan_from_series(series, first_or_last='first'):
    """
    'first' or 'last' to use first or last value in group to fill
    """
    index = _parse_first_or_last_to_index(first_or_last)

    return series[pd.notnull(series)].iloc[index]


def _parse_first_or_last_to_index(first_or_last):
    if first_or_last == 'first':
        return 0
    elif first_or_last == 'last':
        return -1
    else:
        raise ValueError("Did not pass 'first' or 'last'")


def _drop_duplicates(df, byvars):
    """
    Note: inplace
    """
    df.drop_duplicates(subset=byvars, inplace=True)