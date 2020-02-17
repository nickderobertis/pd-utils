import warnings

import numpy as np
import pandas as pd

from pd_utils.merge import groupby_merge
from pd_utils.utils import _to_list_if_str


def long_to_wide(df, groupvars, values, colindex=None, colindex_only=False):
    """

    groupvars = string or list of variables which signify unique observations in the output dataset
    values = string or list of variables which contain the values which need to be transposed
    colindex = string of column or list of strings of columns containing extension for column name
               in the output dataset. If not specified, just uses the
               count of the row within the group. If a list is provided, each column value will be appended
               in order separated by _
    colindex_only = boolean. If true, column names in output data will be only the colindex, and will not
                    include the name of the values variable. Only valid when passing a single value, otherwise
                    multiple columns would have the same name.


    NOTE: Don't have any variables named key or idx

    For example, if we had a long dataset of returns, with returns 12, 24, 36, 48, and 60 months after the date:
            ticker    ret    months
            AA        .01    12
            AA        .15    24
            AA        .21    36
            AA       -.10    48
            AA        .22    60
    and we want to get this to one observation per ticker:
            ticker    ret12    ret24    ret36    ret48    ret60
            AA        .01      .15      .21     -.10      .22
    We would use:
    long_to_wide(df, groupvars='ticker', values='ret', colindex='months')
    """

    df = df.copy()  # don't overwrite original

    # Check for duplicates
    if df.duplicated().any():
        df.drop_duplicates(inplace=True)
        warnings.warn("Found duplicate rows and deleted.")

    # Ensure type of groupvars is correct
    if isinstance(groupvars, str):
        groupvars = [groupvars]
    assert isinstance(groupvars, list)

    # Ensure type of values is correct
    if isinstance(values, str):
        values = [values]
    assert isinstance(values, list)

    if colindex_only and len(values) > 1:
        raise NotImplementedError(
            "set colindex_only to False when passing more than one value"
        )

    # Fixes for colindex
    # Use count of the row within the group for column index if not specified
    if colindex == None:
        df["__idx__"] = df.groupby(groupvars).cumcount()
        colindex = "__idx__"
    # If multiple columns are provided for colindex, combine and drop old cols
    if isinstance(colindex, list):
        df["__idx__"] = ""
        for col in colindex:
            df["__idx__"] = df["__idx__"] + "_" + df[col].astype(str)
            df.drop(col, axis=1, inplace=True)
        colindex = "__idx__"

    df["__key__"] = df[groupvars[0]].astype(str)  # create key variable
    if len(groupvars) > 1:  # if there are multiple groupvars, combine into one key
        for var in groupvars[1:]:
            df["__key__"] = df["__key__"] + "_" + df[var].astype(str)

    # Create seperate wide datasets for each value variable then merge them together
    for i, value in enumerate(values):
        if i == 0:
            combined = df.copy()
        # Create wide dataset
        raw_wide = df.pivot(index="__key__", columns=colindex, values=value)
        if not colindex_only:
            # add value name
            raw_wide.columns = [value + str(col) for col in raw_wide.columns]
        else:
            # remove _ from colindex name
            raw_wide.columns = [str(col).strip("_") for col in raw_wide.columns]
        wide = raw_wide.reset_index()

        # Merge back to original dataset
        combined = combined.merge(wide, how="left", on="__key__")

    return (
        combined.drop([colindex, "__key__"] + values, axis=1)
        .drop_duplicates()
        .reset_index(drop=True)
    )


def averages(df, avgvars, byvars, wtvar=None, count=False, flatten=True):
    """
    Returns equal- and value-weighted averages of variables within groups

    avgvars: List of strings or string of variable names to take averages of
    byvars: List of strings or string of variable names for by groups
    wtvar: String of variable to use for calculating weights in weighted average
    count: False or string of variable name, pass variable name to get count of non-missing
           of that variable within groups.
    flatten: Boolean, False to return df with multi-level index
    """
    # Check types
    assert isinstance(df, pd.DataFrame)
    if isinstance(avgvars, str):
        avgvars = [avgvars]
    else:
        assert isinstance(avgvars, list)
        avgvars = avgvars.copy()  # don't modify existing avgvars inplace
    assert isinstance(byvars, (str, list))
    if wtvar != None:
        assert isinstance(wtvar, str)

    df = df.copy()

    if count:
        df = groupby_merge(df, byvars, "count", subset=count)
        avgvars += [count + "_count"]

    g = df.groupby(byvars)
    avg_df = g.mean()[avgvars]

    if wtvar == None:
        if flatten:
            return avg_df.reset_index()
        else:
            return avg_df

    for var in avgvars:
        colname = var + "_wavg"
        df[colname] = df[wtvar] / g[wtvar].transform("sum") * df[var]

    wavg_cols = [col for col in df.columns if col[-4:] == "wavg"]

    g = df.groupby(byvars)  # recreate because we not have _wavg cols in df
    wavg_df = g.sum()[wavg_cols]

    outdf = pd.concat([avg_df, wavg_df], axis=1)

    if flatten:
        return outdf.reset_index()
    else:
        return outdf


def winsorize(df, pct, subset=None, byvars=None, bot=True, top=True):
    """
    Finds observations above the pct percentile and replaces the with the pct percentile value.
    Does this for all columns, or the subset given by subset

    Required inputs:
    df: Pandas dataframe
    pct: 0 < float < 1 or list of two values 0 < float < 1. If two values are given, the first
         will be used for the bottom percentile and the second will be used for the top. If one value
         is given and both bot and top are True, will use the same value for both.

    Optional inputs:
    subset: List of strings or string of column name(s) to winsorize
    byvars: str, list of strs, or None. Column names of columns identifying groups in the data.
            Winsorizing will be done within those groups.
    bot: bool, True to winsorize bottom observations
    top: bool, True to winsorize top observations

    Example usage:
    winsorize(df, .05, subset='RET') #replaces observations of RET below the 5% and above the 95% values
    winsorize(df, [.05, .1], subset='RET') #replaces observations of RET below the 5% and above the 90% values

    """

    # Check inputs
    assert any([bot, top])  # must winsorize something
    if isinstance(pct, float):
        bot_pct = pct
        top_pct = 1 - pct
    elif isinstance(pct, list):
        bot_pct = pct[0]
        top_pct = 1 - pct[1]
    else:
        raise ValueError("pct must be float or a list of two floats")

    def temp_winsor(col):
        return _winsorize(col, top_pct, bot_pct, top=top, bot=bot)

    # Save column order
    cols = df.columns

    # Get a dataframe of data to be winsorized, and a dataframe of the other columns
    to_winsor, rest = _select_numeric_or_subset(df, subset, extra_include=byvars)

    # Now winsorize
    if byvars:  # use groupby to process groups individually
        to_winsor = groupby_merge(
            to_winsor, byvars, "transform", (temp_winsor), replace=True
        )
    else:  # do entire df, one column at a time
        to_winsor.apply(temp_winsor, axis=0)

    return pd.concat([to_winsor, rest], axis=1)[cols]


def _winsorize(col, top_pct, bot_pct, top=True, bot=True):
    """
    Winsorizes a pandas Series
    """
    if top:
        top_val = col.quantile(top_pct)
        col.loc[col > top_val] = top_val
    if bot:
        bot_val = col.quantile(bot_pct)
        col.loc[col < bot_val] = bot_val
    return col


def _select_numeric_or_subset(df, subset, extra_include=None):
    """
    If subset is not None, selects all numeric columns. Else selects subset.
    If extra_include is not None and subset is None, will select all numeric columns plus
    those in extra_include.
    Returns a tuple of (dataframe containing subset columns, dataframe of other columns)
    """
    if subset == None:
        to_winsor = df.select_dtypes(include=[np.number, np.int64]).copy()
        subset = to_winsor.columns
        rest = df.select_dtypes(exclude=[np.number, np.int64]).copy()
    else:
        if isinstance(subset, str):
            subset = [subset]
        assert isinstance(subset, list)
        to_winsor = df[subset].copy()
        other_cols = [col for col in df.columns if col not in subset]
        rest = df[other_cols].copy()
    if extra_include:
        to_winsor = pd.concat([to_winsor, df[extra_include]], axis=1)
        rest.drop(extra_include, axis=1, inplace=True)

    return (to_winsor, rest)


def var_change_by_groups(df, var, byvars, datevar="Date", numlags=1):
    """
    Used for getting variable changes over time within bygroups.

    NOTE: Dataset is not sorted in this process. Sort the data in the order in which you wish
          lags to be created before running this command.

    Required inputs:
    df: pandas dataframe containing bygroups, a date variable, and variables of interest
    var: str or list of strs, column names of variables to get changes
    byvars: str or list of strs, column names of variables identifying by groups

    Optional inputs:
    datevar: str ot list of strs, column names of variables identifying periods
    numlags: int, number of periods to go back to get change
    """
    var, byvars, datevar = [
        _to_list_if_str(v) for v in [var, byvars, datevar]
    ]  # convert to lists
    short_df = df.loc[
        ~pd.isnull(df[byvars]).any(axis=1), var + byvars + datevar
    ].drop_duplicates()
    for v in var:
        short_df[v + "_lag"] = short_df.groupby(byvars)[v].shift(numlags)
        short_df[v + "_change"] = short_df[v] - short_df[v + "_lag"]
    dropvars = [v for v in var] + [v + "_lag" for v in var]
    short_df = short_df.drop(dropvars, axis=1)
    return df.merge(short_df, on=datevar + byvars, how="left")


def state_abbrev(df, col, toabbrev=False):
    df = df.copy()
    states_to_abbrev = {
        "Alabama": "AL",
        "Montana": "MT",
        "Alaska": "AK",
        "Nebraska": "NE",
        "Arizona": "AZ",
        "Nevada": "NV",
        "Arkansas": "AR",
        "New Hampshire": "NH",
        "California": "CA",
        "New Jersey": "NJ",
        "Colorado": "CO",
        "New Mexico": "NM",
        "Connecticut": "CT",
        "New York": "NY",
        "Delaware": "DE",
        "North Carolina": "NC",
        "Florida": "FL",
        "North Dakota": "ND",
        "Georgia": "GA",
        "Ohio": "OH",
        "Hawaii": "HI",
        "Oklahoma": "OK",
        "Idaho": "ID",
        "Oregon": "OR",
        "Illinois": "IL",
        "Pennsylvania": "PA",
        "Indiana": "IN",
        "Rhode Island": "RI",
        "Iowa": "IA",
        "South Carolina": "SC",
        "Kansas": "KS",
        "South Dakota": "SD",
        "Kentucky": "KY",
        "Tennessee": "TN",
        "Louisiana": "LA",
        "Texas": "TX",
        "Maine": "ME",
        "Utah": "UT",
        "Maryland": "MD",
        "Vermont": "VT",
        "Massachusetts": "MA",
        "Virginia": "VA",
        "Michigan": "MI",
        "Washington": "WA",
        "Minnesota": "MN",
        "West Virginia": "WV",
        "Mississippi": "MS",
        "Wisconsin": "WI",
        "Missouri": "MO",
        "Wyoming": "WY",
    }
    if toabbrev:
        df[col] = df[col].replace(states_to_abbrev)
    else:
        abbrev_to_states = dict((v, k) for k, v in states_to_abbrev.items())
        df[col] = df[col].replace(abbrev_to_states)

    return df


def _join_col_strings(*args):
    strs = [str(arg) for arg in args]
    return "_".join(strs)


def join_col_strings(df, cols):
    """
    Takes a dataframe and column name(s) and concatenates string versions of the columns with those names.
    Useful for when a group is identified by several variables and we need one key variable to describe a group.
    Returns a pandas Series.

    Required inputs:
    df: pandas dataframe
    cols: str or list, names of columns in df to be concatenated
    """

    if isinstance(cols, str):
        cols = [cols]
    assert isinstance(cols, list)

    jc = np.vectorize(_join_col_strings)

    return pd.Series(jc(*[df[col] for col in cols]))