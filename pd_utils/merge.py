import functools
import timeit
import datetime
from typing import List, Dict, Union, Optional, Callable

import numpy as np
import pandas as pd
from numpy import nan

from pd_utils.timer import estimate_time
from pd_utils.query import sql


def groupby_merge(df, byvars: Union[str, List[str]], func_str: str, *func_args, subset: Union[str, List[str]] = "all",
                  replace: bool = False):
    """
    Creates a pandas groupby object, applies the aggregation function in func_str, and merges back the
    aggregated data to the original dataframe.


    :param df:
    :param byvars: column names which uniquely identify groups
    :param func_str: name of groupby aggregation function such as 'min', 'max', 'sum', 'count', etc.
    :param func_args: arguments to pass to func
    :param subset: column names for which to apply aggregation functions or 'all' for all columns
    :param replace: True to replace original columns in the data with aggregated/transformed columns
    :return:

    :Example:

    >>> import pd_utils
    >>> df = pd_utils.groupby_merge(df, ['PERMNO','byvar'], 'max', subset='RET')
    """

    # Convert byvars to list if neceessary
    if isinstance(byvars, str):
        byvars = [byvars]

    # Store all variables except byvar in subset if subset is 'all'
    if subset == "all":
        subset = [col for col in df.columns if col not in byvars]

    # Convert subset to list if necessary
    if isinstance(subset, str):
        subset = [subset]

    # Groupby expects to receive a string if there is a single variable
    groupby_subset: Union[str, List[str]]
    if len(subset) == 1:
        groupby_subset = subset[0]
    else:
        groupby_subset = subset

    if func_str == "transform":
        # transform works very differently from other aggregation functions

        # First we need to deal with nans in the by variables. If there are any nans, transform will error out
        # Therefore we must fill the nans in the by variables beforehand and replace afterwards
        df[byvars] = df[byvars].fillna(value="__tempnan__")

        # Now we must deal with nans in the subset variables. If there are any nans, tranform will error out
        # because it tries to ignore the nan. Therefore we must remove these rows from the dataframe,
        # transform, then add those rows back.
        any_nan_subset_mask = pd.Series(
            [all(i) for i in (zip(*[~pd.isnull(df[col]) for col in subset]))],
            index=df.index,
        )
        no_nans = df[any_nan_subset_mask]

        grouped = no_nans.groupby(byvars)
        func = getattr(
            grouped, func_str
        )  # pull method of groupby class with same name as func_str
        grouped = func(*func_args)[
            groupby_subset
        ]  # apply the class method and select subset columns
        if isinstance(grouped, pd.DataFrame):
            grouped.columns = [
                col + "_" + func_str for col in grouped.columns
            ]  # rename transformed columns
        elif isinstance(grouped, pd.Series):
            grouped.name = str(grouped.name) + "_" + func_str

        df.replace("__tempnan__", nan, inplace=True)  # fill nan back into dataframe

        # Put nan rows back
        grouped = grouped.reindex(df.index)

        full = pd.concat([df, grouped], axis=1)

    else:  # .min(), .max(), etc.

        #         grouped = df.groupby(byvars, as_index=False)[byvars + subset]
        #         func = getattr(grouped, func_str) #pull method of groupby class with same name as func_str
        #         grouped = func(*func_args) #apply the class method

        grouped = df.groupby(byvars)[groupby_subset]
        func = getattr(
            grouped, func_str
        )  # pull method of groupby class with same name as func_str
        grouped = func(*func_args)  # apply the class method
        grouped = grouped.reset_index()

        # Merge and output
        full = df.merge(grouped, how="left", on=byvars, suffixes=["", "_" + func_str])

    if replace:
        _replace_with_transformed(full, func_str)

    return full


def _replace_with_transformed(df, func_str="transform"):
    transform_cols = [col for col in df.columns if col.endswith("_" + func_str)]
    orig_names = [col[: col.find("_" + func_str)] for col in transform_cols]
    df.drop(orig_names, axis=1, inplace=True)
    df.rename(
        columns={old: new for old, new in zip(transform_cols, orig_names)}, inplace=True
    )


def groupby_index(df: pd.DataFrame, byvars: Union[str, List[str]], sortvars: Optional[Union[str, List[str]]] = None,
                  ascending: bool = True):
    """
    Returns a dataframe which is a copy of the old one with an additional column containing an index
    by groups. Each time the bygroup changes, the index restarts at 0.

    :param df:
    :param byvars: column names containing group identifiers
    :param sortvars: column names to sort by within by groups
    :param ascending: direction of sort
    :return:
    """

    # Convert sortvars and byvars to list if necessary
    if isinstance(sortvars, str):
        sortvars = [sortvars]
    if sortvars is None:
        sortvars = []
    if isinstance(byvars, str):
        byvars = [byvars]

    df = df.copy()  # don't modify the original dataframe
    df.sort_values(byvars + sortvars, inplace=True, ascending=ascending)
    df["__temp_cons__"] = 1
    df = groupby_merge(
        df,
        byvars,
        "transform",
        (lambda x: [i for i in range(len(x))]),
        subset=["__temp_cons__"],
    )
    df.drop("__temp_cons__", axis=1, inplace=True)
    return df.rename(columns={"__temp_cons___transform": "group_index"})


def apply_func_to_unique_and_merge(series: pd.Series, func: Callable) -> pd.Series:
    """
    This function reduces the given series down to unique values, applies the function,
    then expands back up to the original shape of the data.

    Many Pandas functions can be slow because they're doing repeated work. This can help
    optimize some operations.

    :param series:
    :param func: function to be applied to the series
    :return:

    :Usage:

    >>>import functools
    >>>to_datetime = functools.partial(pd.to_datetime, format='%Y%m')
    >>>apply_func_to_unique_and_merge(df['MONTH'], to_datetime)

    """
    unique = pd.Series(series.unique())
    new = unique.apply(func)

    for_merge = pd.concat([unique, new], axis=1)
    num_cols = [i for i in range(len(for_merge.columns) - 1)]  # names of new columns
    for_merge.columns = [series.name] + num_cols

    orig_df = pd.DataFrame(series)
    orig_df.reset_index(inplace=True)

    return (
        for_merge.merge(orig_df, how="right", on=[series.name])
        .sort_values("index")
        .reset_index()
        .loc[:, num_cols]
    )


def left_merge_latest(
    df: pd.DataFrame,
    df2: pd.DataFrame,
    on: Union[str, List[str]],
    left_datevar: str = "Date",
    right_datevar: str = "Date",
    max_offset: Optional[Union[int, datetime.timedelta]] = None,
    backend: str = "pandas",
    low_memory: bool = False,
):
    """
    Left merges df2 to df using on, but grabbing the most recent observation (right_datevar will be
    the soonest earlier than left_datevar). Useful for situations where data needs to be merged with
    mismatched dates, and just the most recent data available is needed.

    :param df: Pandas dataframe containing source data (all rows will be kept), must have on variables
        and left_datevar
    :param df2: Pandas dataframe containing data to be merged (only the most recent rows before source
        data will be kept)
    :param on: names of columns on which to match, excluding date
    :param left_datevar: name of date variable on which to merge in df
    :param right_datevar: name of date variable on which to merge in df2
    :param max_offset: maximum amount of time to go back to look for a match.
        When datevar is a datetime column, pass datetime.timedelta. When datevar is an int column
        (e.g. year), pass an int. Currently only applicable for backend 'pandas'
    :param backend: 'pandas' or 'sql'. Specify the underlying machinery used to perform the merge.
             'pandas' means native pandas, while 'sql' uses pandasql. Try 'sql' if you run
             out of memory.
    :param low_memory: True to reduce memory usage but decrease calculation speed
    :return:
    """
    if isinstance(on, str):
        on = [on]

    if backend.lower() in ("pandas", "pd"):
        if low_memory:
            return _left_merge_latest_pandas_low_memory(
                df, df2, on, left_datevar=left_datevar, right_datevar=right_datevar
            )
        return _left_merge_latest_pandas(
            df,
            df2,
            on,
            left_datevar=left_datevar,
            right_datevar=right_datevar,
            max_offset=max_offset,
        )
    elif backend.lower() in ("sql", "pandasql"):
        if max_offset is not None:
            raise NotImplementedError("cannot yet handle max_offset for sql backend")
        return _left_merge_latest_sql(
            df, df2, on, left_datevar=left_datevar, right_datevar=right_datevar
        )
    else:
        raise ValueError("select backend='pandas' or backend='sql'.")


def _left_merge_latest_pandas(
    df, df2, on, left_datevar="Date", right_datevar="Date", max_offset=None
):
    many = df.loc[:, on + [left_datevar]].merge(df2, on=on, how="left")

    rename = False
    # if they are named the same, pandas will automatically add _x and _y to names
    if left_datevar == right_datevar:
        rename = True  # will need to rename the _x datevar for the last step
        orig_left_datevar = left_datevar
        left_datevar += "_x"
        right_datevar += "_y"

    lt = many.loc[
        many[left_datevar] >= many[right_datevar]
    ]  # left with datadates less than date

    if max_offset is not None:
        lt = lt.loc[lt[right_datevar] >= lt[left_datevar] - max_offset]

    # find rows within groups which have the maximum right_datevar (soonest before left_datevar)
    data_rows = (
        lt.groupby(on + [left_datevar])[right_datevar]
        .max()
        .reset_index()
        .merge(lt, on=on + [left_datevar, right_datevar], how="left")
    )

    if rename:  # remove the _x for final merge
        data_rows.rename(columns={left_datevar: orig_left_datevar}, inplace=True)
        datevar_for_merge = orig_left_datevar
    else:
        datevar_for_merge = left_datevar

    merged = df.merge(data_rows, on=on + [datevar_for_merge], how="left")
    # for some reason is getting converted to object type
    merged[right_datevar] = pd.to_datetime(merged[right_datevar])

    return merged


def _left_merge_latest_pandas_low_memory(
    df: pd.DataFrame,
    df2: pd.DataFrame,
    on: List[str],
    left_datevar="Date",
    right_datevar="Date",
):

    MERGE_DATE = "_merge_date"

    def _get_latest_date(orig_date, dates=None):
        if dates is None:
            dates = []
        last_date = None
        for date in dates:
            if date > orig_date:
                return last_date
            last_date = date
        # Did not find any dates greater than passed date
        last_date = max(dates)
        if last_date < orig_date:
            return last_date

    def _to_datetime(date_like):
        """
        Skips converting NaT and NaN but does convert dates
        Args:
            date_like:

        Returns:

        """
        if pd.isnull(date_like):
            return date_like

        return pd.to_datetime(date_like)

    # Need to handle conversion to datetime for created match date if originally a datetime type
    date_is_datetime_type = (
        left_datevar in df.select_dtypes(include=[np.datetime64]).columns
    )

    dfs_for_concat = []
    df2_for_slice = df2[on + [right_datevar]].set_index(on)
    grouped = df.groupby(on)
    num_grouped = len(grouped)
    count = -1
    start_time = timeit.default_timer()
    print(
        "Starting low memory handling for left_merge_latest. Processing groups one at a time.\n"
    )
    for group_on, group_df in grouped:
        count += 1
        try:
            df2_group_df = df2_for_slice.loc[group_on]
        except KeyError:
            # Did not find this obs in the second df, cannot get date to merge
            group_df[MERGE_DATE] = np.nan
            dfs_for_concat.append(group_df)
            continue
        if isinstance(df2_group_df, pd.Series):
            # Only got a single row for group df, so got a single date, wrap in a list
            df2_dates = [df2_group_df[right_datevar]]
        else:
            df2_dates = df2_group_df[right_datevar].dropna().unique()
            df2_dates.sort()
        get_latest_date = functools.partial(_get_latest_date, dates=df2_dates)
        group_df[MERGE_DATE] = group_df[left_datevar].apply(get_latest_date)
        dfs_for_concat.append(group_df)
        estimate_time(num_grouped, count, start_time)
    print("\nFinished processing groups to get latest date. Now doing final merge.")

    df_for_merge = pd.concat(dfs_for_concat, axis=0)
    del dfs_for_concat  # free up memory

    if date_is_datetime_type:
        df_for_merge[MERGE_DATE] = df_for_merge[MERGE_DATE].apply(_to_datetime)
    else:
        # Handle other type conversions
        desired_dtype = df_for_merge[left_datevar].dtype
        df_for_merge[MERGE_DATE] = df_for_merge[MERGE_DATE].astype(desired_dtype)

    merged = df_for_merge.merge(
        df2, how="left", left_on=on + [MERGE_DATE], right_on=on + [right_datevar]
    )
    merged.drop(MERGE_DATE, axis=1, inplace=True)

    rename = False
    # if they are named the same, pandas will automatically add _x and _y to names
    if left_datevar == right_datevar:
        rename = True  # will need to rename the _x datevar for the last step
        orig_left_datevar = left_datevar
        left_datevar += "_x"
        right_datevar += "_y"

    if rename:  # remove the _x
        merged.rename(columns={left_datevar: orig_left_datevar}, inplace=True)

    return merged


def _left_merge_latest_sql(df, df2, on, left_datevar="Date", right_datevar="Date"):

    if left_datevar == right_datevar:
        df2 = df2.copy()
        df2.rename(columns={right_datevar: right_datevar + "_y"}, inplace=True)
        right_datevar += "_y"

    # Pandasql cannot handle spaces in names, replace
    df_reverse_col_replacements = _replace_df_columns_char(
        df, find_str=" ", replace_str="_"
    )
    df2_reverse_col_replacements = _replace_df_columns_char(
        df2, find_str=" ", replace_str="_"
    )
    orig_on = on.copy()
    on = [col.replace(" ", "_") for col in on]

    on_str = " and \n    ".join(["a.{0} = b.{0}".format(i) for i in on])
    groupby_str = ", ".join(on)
    a_cols = ", ".join(["a." + col for col in on + [left_datevar]])
    b_cols = ", ".join(["b." + col for col in df2.columns if col not in on])
    query = """
    select {5}, {4}
    from
        df a
    left join
        df2 b
    on
        {0} and
        a.{1} >= b.{2}
    group by a.{3}, a.{1}
    having
        b.{2} = max(b.{2})
    """.format(
        on_str, left_datevar, right_datevar, groupby_str, b_cols, a_cols
    )

    result = sql([df, df2], query)

    # Reverse the name replacements, bring spaces back
    df.rename(columns=df_reverse_col_replacements, inplace=True)
    df2.rename(columns=df2_reverse_col_replacements, inplace=True)
    # Get all replacements, for result table which may have columns from both
    df_reverse_col_replacements.update(df2_reverse_col_replacements)
    result.rename(columns=df_reverse_col_replacements, inplace=True)

    merged = df.merge(result, on=orig_on + [left_datevar], how="left")
    # for some reason is getting converted to object type
    merged[right_datevar] = pd.to_datetime(merged[right_datevar])

    return merged


def _replace_df_columns_char(
    df: pd.DataFrame, find_str: str = " ", replace_str: str = "_"
) -> Dict[str, str]:
    """
    Note: inplace

    Args:
        df:
        find_str:
        replace_str:

    Returns: reverse replacement dict, use to reverse changes

    """
    replace_dict = {}
    for col in df.columns:
        replace_dict[col] = col.replace(find_str, replace_str)

    df.rename(columns=replace_dict, inplace=True)

    reverse_replace_dict = {v: k for k, v in replace_dict.items()}
    return reverse_replace_dict