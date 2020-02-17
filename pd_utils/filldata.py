import pandas as pd
import numpy as np
from functools import partial
from itertools import product
from typing import List, Optional

from pd_utils.utils import _to_list_if_str, _to_series_if_str, _to_name_if_series


def fillna_by_groups_and_keep_one_per_group(
    df, byvars, exclude_cols=None, str_vars="first", num_vars="mean"
):
    """
    Fills missing values by group, with different handling for string variables versus numeric,
    then keeps one observation per group.

    WARNING: do not use if index is important, it will be dropped
    """
    byvars = _to_list_if_str(byvars)
    if exclude_cols:
        exclude_cols = _to_list_if_str(exclude_cols)

    df = fillna_by_groups(
        df, byvars, exclude_cols=exclude_cols, str_vars=str_vars, num_vars=num_vars
    )
    _drop_duplicates(df, byvars)

    return df


def fillna_by_groups(df, byvars, exclude_cols=None, str_vars="first", num_vars="mean"):
    """
    Fills missing values by group, with different handling for string variables versus numeric

    WARNING: do not use if index is important, it will be dropped
    """
    byvars = _to_list_if_str(byvars)

    if exclude_cols:
        cols_to_fill = [
            col
            for col in df.columns
            if (col not in exclude_cols) and (col not in byvars)
        ]
        concat_vars = byvars + exclude_cols
    else:
        cols_to_fill = [col for col in df.columns if col not in byvars]
        concat_vars = byvars

    _fill_data = partial(_fill_data_for_series, str_vars=str_vars, num_vars=num_vars)

    out_dfs = []
    for group, group_df in df[byvars + cols_to_fill].groupby(byvars, as_index=False):
        out_dfs.append(group_df.apply(_fill_data, axis=0))

    filled = pd.concat(out_dfs, axis=0).reset_index(drop=True)

    filled = _restore_nans_after_fill(
        filled
    )  # _fill_data places -999.999 in place of nans, now convert back

    return filled


def add_missing_group_rows(
    df,
    group_id_cols: List[str],
    non_group_id_cols: List[str],
    fill_method: Optional[str] = "ffill",
    fill_limit: Optional[int] = None,
):
    """
    Adds rows so that each group has all non group IDs, optionally filling values by a pandas fill method

    :param df:
    :param group_id_cols: typically entity ids. these ids represents groups in the data. data will not be
            forward/back filled across differences in these ids.
    :param non_group_id_cols: typically date or time ids. data will be forward/back filled across differences in these ids
    :param fill_method: pandas fill methods, None to not fill
    :param fill_limit: pandas fill limit
    :return:
    """
    fill_id_cols = group_id_cols + non_group_id_cols
    fill_ids = [df[fill_id_col].unique() for fill_id_col in fill_id_cols]
    index_df = pd.DataFrame([i for i in product(*fill_ids)], columns=fill_id_cols)

    merged = index_df.merge(df, how="left", on=fill_id_cols)

    # Newly created rows will have missing values. Sort and fill
    merged.sort_values(fill_id_cols, inplace=True)
    # TODO [#3]: Update add_missing_group_rows to not fill nans in existing data
    #
    # this method can still fill nans in existing data, not just created rows

    # if fill_method is None, don't call fillna at all, return with NaNs
    if fill_method is not None:
        merged = merged.groupby(group_id_cols, as_index=False).fillna(
            method=fill_method, limit=fill_limit
        )

    return merged


def drop_missing_group_rows(df, fill_id_cols):
    drop_subset = [col for col in df.columns if col not in fill_id_cols]
    return df.dropna(subset=drop_subset, how="all")


def _fill_data_for_series(series, str_vars="first", num_vars="mean"):
    index = _get_non_nan_value_index(series, str_vars)
    if index is None:
        # All nans, can't do anything but return back nothing
        # But transform ignores nans in the output and then complains when the sizes don't match.
        # So instead, put a placeholder of -999.999
        return pd.Series([-999.999 for i in range(len(series))])
    # handle numeric
    if series.dtype in (np.float64, np.int64):
        if num_vars in ("first", "last"):
            # Overwrite index for that of num vars if not using the same value as for str vars
            if num_vars != str_vars:
                index = _get_non_nan_value_index(series, num_vars)
            return _fill_data_for_str_series(series, non_nan_index=index)
        return _fill_data_for_numeric_series(series, fill_function=num_vars)
    # handle strs
    else:
        return _fill_data_for_str_series(series, non_nan_index=index)


def _fill_data_for_numeric_series(series, fill_function="mean"):
    return series.fillna(series.agg(fill_function))


def _fill_data_for_str_series(series, non_nan_index):
    fill_value = series.loc[non_nan_index]

    return series.fillna(fill_value)


def _get_non_nan_value_index(series, first_or_last):
    if first_or_last == "first":
        return series.first_valid_index()
    elif first_or_last == "last":
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


def fill_excluded_rows(df, byvars, fillvars=None, **fillna_kwargs):
    """
    Takes a dataframe which does not contain all possible combinations of byvars as rows. Creates
    those rows if fillna_kwargs are passed, calls fillna using fillna_kwargs for fillvars

    :param df:
    :param byvars: variables on which dataset should be expanded to product. Can pass a str, list of
            strs, or a list of pd.Series.
    :param fillvars: optional variables to apply fillna to
    :param fillna_kwargs: See pandas.DataFrame.fillna for kwargs, value=0 is common
    :return:

    :Example:

    An example::

        df:
                     date     id  var
            0  2003-06-09 42223C    1
            1  2003-06-10 09255G    2

        with fillna_for_excluded_rows(df, byvars=['date','id'], fillvars='var', value=0) becomes:

                      date     id  var
            0  2003-06-09 42223C    1
            1  2003-06-10 42223C    0
            2  2003-06-09 09255G    0
            3  2003-06-10 09255G    2
    """
    byvars, fillvars = [
        _to_list_if_str(v) for v in [byvars, fillvars]
    ]  # convert to lists

    #     multiindex = [df[i].dropna().unique() for i in byvars]
    multiindex = [_to_series_if_str(df, i).dropna().unique() for i in byvars]
    byvars = [_to_name_if_series(i) for i in byvars]  # get name of any series

    all_df = pd.DataFrame(index=pd.MultiIndex.from_product(multiindex)).reset_index()
    all_df.columns = byvars
    merged = all_df.merge(df, how="left", on=byvars)

    if fillna_kwargs:
        fillna_kwargs.update({"inplace": False})
        merged[fillvars] = merged[fillvars].fillna(**fillna_kwargs)
    return merged