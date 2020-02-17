import functools
import time
import timeit
import warnings
from functools import partial
from multiprocessing import Pool
from typing import Optional, List, Union, Tuple

import numpy as np
import pandas as pd

from pd_utils.timer import estimate_time
from pd_utils.transform import averages


def portfolio(
    df: pd.DataFrame,
    groupvar: str,
    ngroups: int = 10,
    cutoffs: Optional[List[Union[float, int]]] = None,
    quant_cutoffs: Optional[List[float]] = None,
    byvars: Optional[Union[str, List[str]]] = None,
    cutdf: Optional[pd.DataFrame] = None,
    portvar: str = "portfolio",
    multiprocess: bool = False,
):
    """
    Constructs portfolios based on percentile values of groupvar.

    If ngroups=10, then will form 10 portfolios,
    with portfolio 1 having the bottom 10 percentile of groupvar, and portfolio 10 having the top 10 percentile
    of groupvar.

    :Notes:

    * Resets index and drops in output data, so don't use if index is important (input data not affected)
    * If using a cutdf, MUST have the same bygroups as df. The number of observations within each bygroup
      can be different, but there MUST be a one-to-one match of bygroups, or this will NOT work correctly.
      This may require some cleaning of the cutdf first.
    * For some reason, multiprocessing seems to be slower in testing, so it is disabled by default

    :param df: input data
    :param groupvar: name of variable in df to form portfolios on
    :param ngroups: number of portfolios to form. will be ignored if option cutoffs or quant_cutoffs is passed
    :param cutoffs: e.g. [100, 10000] to form three portfolios, 1 would be < 100, 2 would be > 100 and < 10000,
        3 would be > 10000. cannot be used with option ngroups
    :param quant_cutoffs: eg. [0.1, 0.9] to form three portfolios. 1 would be lowest 10% of data,
        2 would be > 10 and < 90 percentiles, 3 would be highest 10%. All will be within byvars if byvars are passed
    :param byvars: name of variable(s) in df, finds portfolios within byvars. For example if byvars='Month',
        would take each month and form portfolios based on the percentiles of the groupvar during only that month
    :param cutdf: optionally determine percentiles using another dataset. See second note.
    :param portvar: name of portfolio variable in the output dataset
    :param multiprocess: set to True to use all available processors,
        set to False to use only one, pass an int less or equal to than number of
        processors to use that amount of processors
    :return:
    """
    # Check types
    _check_portfolio_inputs(
        df,
        groupvar,
        ngroups=ngroups,
        byvars=byvars,
        cutdf=cutdf,
        portvar=portvar,
        cutoffs=cutoffs,
        quant_cutoffs=quant_cutoffs,
    )
    byvars = _assert_byvars_list(byvars)
    if cutdf != None:
        assert isinstance(cutdf, pd.DataFrame)
    else:  # this is where cutdf == None, the default case
        cutdf = df
        tempcutdf = cutdf.copy()

    # With passed cutoffs, can skip all logic to calculate cutoffs, and go right to portfolio sort
    if cutoffs is not None:
        # Must add a minimum and maximum to cutoffs (bottom of lowest port, top of highest port)
        # for code to work properly
        min_groupvar_value = df[groupvar].min()
        max_groupvar_value = df[groupvar].max()
        all_cutoffs = [min_groupvar_value] + cutoffs + [max_groupvar_value]
        return _sort_into_ports(df, all_cutoffs, portvar, groupvar)

    # Hard cutoffs not passed, handle percentile based portfolios (either ngroups or passed quant_cuts)
    if quant_cutoffs is not None:
        # Must add a minimum and maximum to quant cutoffs and scale to 0-100 for code to work properly
        percentiles = [0, *[q * 100 for q in quant_cutoffs], 100]
    else:
        # ngroups handling
        pct_per_group = 100 / ngroups
        percentiles = [
            i * pct_per_group for i in range(ngroups)
        ]  # percentile values, e.g. 0, 10, 20, 30... 100
        percentiles += [100]

    # Create new functions with common arguments added. First function is for handling entire df, second is for
    # splitting into numpy arrays by byvars
    create_cutoffs_if_necessary_and_sort_into_ports = functools.partial(
        _create_cutoffs_and_sort_into_ports,
        groupvar=groupvar,
        portvar=portvar,
        percentiles=percentiles,
    )

    sort_arr_list_into_ports_and_return_series = functools.partial(
        _sort_arr_list_into_ports_and_return_series,
        percentiles=percentiles,
        multiprocess=multiprocess,
    )

    split = functools.partial(_split, keepvars=[groupvar], force_numeric=True)
    tempdf = df.copy()

    # If there are no byvars, just complete portfolio sort
    if byvars is None:
        return create_cutoffs_if_necessary_and_sort_into_ports(tempdf, cutdf)

    # The below rename is incase there is already a variable named index in the data
    # The rename will just not do anything if there's not
    tempdf = (
        tempdf.reset_index(drop=True)
        .rename(columns={"index": "__temp_index__"})
        .reset_index()
    )  # get a variable 'index' containing obs count

    # Also replace index in byvars if there
    temp_byvars = [b if b != "index" else "__temp_index__" for b in byvars]
    all_byvars = [temp_byvars, byvars]  # list of lists

    # else, deal with byvars
    # First create a key variable based on all the byvars
    for i, this_df in enumerate([tempdf, tempcutdf]):
        this_df["__key_var__"] = "key"  # container for key
        for col in [this_df[c].astype(str) for c in all_byvars[i]]:
            this_df["__key_var__"] += col
        this_df.sort_values("__key_var__", inplace=True)

    # Now split into list of arrays and process
    array_list = split(tempdf)
    cut_array_list = split(tempcutdf)

    tempdf = tempdf.reset_index(
        drop=True
    )  # need to reset index again for adding new column
    tempdf[portvar] = sort_arr_list_into_ports_and_return_series(
        array_list, cut_array_list
    )
    return (
        tempdf.sort_values("index")
        .drop(["__key_var__", "index"], axis=1)
        .rename(columns={"__temp_index__": "index"})
        .reset_index(drop=True)
    )


def portfolio_averages(
    df: pd.DataFrame,
    groupvar: str,
    avgvars: Union[str, List[str]],
    ngroups: int = 10,
    byvars: Optional[Union[str, List[str]]] = None,
    cutdf: pd.DataFrame = None,
    wtvar: Optional[str] = None,
    count: Union[str, bool] = False,
    portvar: str = "portfolio",
    avgonly: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Creates portfolios and calculates equal- and value-weighted averages of variables within portfolios.

    If ngroups=10,
    then will form 10 portfolios, with portfolio 1 having the bottom 10 percentile of groupvar, and portfolio 10 having
    the top 10 percentile of groupvar.

    :Notes:

    Resets index and drops in output data, so don't use if index is important (input data not affected)

    :param df: input data
    :param groupvar: name of variable in df to form portfolios on
    :param avgvars: variables to be averaged
    :param ngroups: number of portfolios to form
    :param byvars: name of variable(s) in df, finds portfolios within byvars. For example if byvars='Month',
            would take each month and form portfolios based on the percentiles of the groupvar during only that month
    :param cutdf: optionally determine percentiles using another dataset
    :param wtvar: name of variable in df to use for weighting in weighted average
    :param count: pass variable name to get count of non-missing of that variable within groups.
    :param portvar: name of portfolio variable in the output dataset
    :param avgonly: True to return only averages, False to return (averages, individual observations with portfolios)
    :return:
    """
    ports = portfolio(
        df, groupvar, ngroups=ngroups, byvars=byvars, cutdf=cutdf, portvar=portvar
    )
    if byvars:
        assert isinstance(byvars, (str, list))
        if isinstance(byvars, str):
            byvars = [byvars]
        by = [portvar] + byvars
        avgs = averages(ports, avgvars, byvars=by, wtvar=wtvar, count=count)
    else:
        avgs = averages(ports, avgvars, byvars=portvar, wtvar=wtvar, count=count)

    if avgonly:
        return avgs
    else:
        return avgs, ports


def long_short_portfolio(df: pd.DataFrame, portvar: str, byvars: Optional[Union[str, List[str]]] = None,
                         retvars: Optional[Union[str, List[str]]] = None, top_minus_bot: bool = True):
    """
    Takes a df with a column of numbered portfolios and creates a new
    portfolio which is long the top portfolio and short the bottom portfolio.

    :param df: dataframe containing a column with portfolio numbers
    :param portvar: name of column containing portfolios
    :param byvars: column names containing groups for portfolios.
        Calculates long-short within these groups. These should be the same groups
        in which portfolios were formed.
    :param retvars: variables to return in the long-short dataset.
        By default, will use all numeric variables in the df
    :param top_minus_bot: True to be long the top portfolio, short the bottom portfolio.
       False to be long the bottom portfolio, short the top portfolio.
    :return: a df of long-short portfolio
    """
    long, short = _select_long_short_ports(df, portvar, top_minus_bot=top_minus_bot)
    return _portfolio_difference(
        df, portvar, long, short, byvars=byvars, retvars=retvars
    )


def _select_long_short_ports(df, portvar, top_minus_bot=True):
    """
    Finds the appropriate portfolio number and returns (long number, short number)
    """
    #Get numbered value of highest and lowest portfolio
    top = max(df[portvar])
    bot = min(df[portvar])

    if top_minus_bot:
        return top, bot
    else:
        return bot, top


def _portfolio_difference(df, portvar, long, short, byvars=None, retvars=None):
    """
    Calculates long portfolio minus short portfolio
    """
    if byvars:
        out = df[df[portvar] == long].set_index(byvars) - df[df[portvar] == short].set_index(byvars)
    else:
        out = df[df[portvar] == long] - df[df[portvar] == short]

    if retvars:
        return out[retvars]
    else:
        return out


def _sort_into_ports(df, cutoffs, portvar, groupvar):
        df[portvar] = 0
        for i, (low_cut, high_cut) in enumerate(zip(cutoffs[:-1],cutoffs[1:])):
                rows = df[(df[groupvar] >= low_cut) & (df[groupvar] <= high_cut)].index
                df.loc[rows, portvar] = i + 1
        return df


def _create_cutoffs(cutdf, groupvar, percentiles):
    return [np.nanpercentile(cutdf[groupvar], i) for i in percentiles]


def _create_cutoffs_and_sort_into_ports(df, cutdf, groupvar, portvar, percentiles):
    cutoffs = _create_cutoffs(cutdf, groupvar, percentiles)
    return _sort_into_ports(df, cutoffs, portvar, groupvar)


def _split(df, keepvars, force_numeric=False):
    """
    Splits a dataframe into a list of arrays based on a key variable. Pass keepvars
    to keep variables other than the key variable
    """
#     df = df.sort_values('__key_var__') #now done outside of function
    small_df = df[['__key_var__'] + keepvars]
    arr = small_df.values
    splits = []
    for i in range(arr.shape[0]):
        if i == 0: continue
        if arr[i,0] != arr[i-1,0]: #different key
            splits.append(i)
    outarr = arr[:,1:]
    if force_numeric:
        outarr = outarr.astype('float64')
    return np.split(outarr, splits)


def _create_cutoffs_arr(arr, percentiles):
    arr = arr[~np.isnan(arr) & ~np.isinf(arr)]
    if arr.size == 0:
        return False
    return [np.percentile(arr, i) for i in percentiles]


def _sort_arr_into_ports(arr, cutoffs):
    port_cutoffs = _gen_port_cutoffs(cutoffs)
    portfolio_match = partial(_portfolio_match, port_cutoffs=port_cutoffs)
    return [portfolio_match(elem) for elem in arr]


def _portfolio_match(elem, port_cutoffs):
    if np.isnan(elem) or np.isinf(elem): return 0
    return [index + 1 for index, bot, top in port_cutoffs \
            if elem >= bot and elem <= top][0]


def _gen_port_cutoffs(cutoffs):
    return [(i, low_cut, high_cut) \
                    for i, (low_cut, high_cut) \
                    in enumerate(zip(cutoffs[:-1],cutoffs[1:]))]


def _sort_arr_list_into_ports(array_list, cut_array_list, percentiles, multiprocess,
                              cutoffs: Optional[List[Union[float, int]]] = None):
    common_args = (
        array_list,
        cut_array_list,
        percentiles
    )
    common_kwargs = dict(
        cutoffs=cutoffs
    )
    if multiprocess:
        if isinstance(multiprocess, int):
            return _sort_arr_list_into_ports_mp(*common_args, mp=multiprocess, **common_kwargs)
        else:
            return _sort_arr_list_into_ports_mp(*common_args, **common_kwargs)
    else:
        return _sort_arr_list_into_ports_sp(*common_args, **common_kwargs)


def _create_cutoffs_arr_if_necessary_and_sort_into_ports(data_tup, percentiles,
                                                         cutoffs: Optional[List[Union[float, int]]] = None):
    arr, cutarr = data_tup
    if cutoffs is None:
        cutoffs = _create_cutoffs_arr(cutarr, percentiles)
    if cutoffs:
        return _sort_arr_into_ports(arr, cutoffs)
    else:
        return [0 for elem in arr]


def _sort_arr_list_into_ports_sp(array_list, cut_array_list, percentiles,
                                 cutoffs: Optional[List[Union[float, int]]] = None):
    outlist = []
    for i, arr in enumerate(array_list):
        result = _create_cutoffs_arr_if_necessary_and_sort_into_ports((arr, cut_array_list[i]),
                                                                      percentiles, cutoffs=cutoffs)
        outlist.append(result)
    return outlist


def _sort_arr_list_into_ports_mp(array_list, cut_array_list, percentiles, mp=None):
    if mp:
        with Pool(mp) as pool: #use mp # processors
            return _sort_arr_list_into_ports_mp_main(array_list, cut_array_list, percentiles, pool)
    else:
        with Pool() as pool: #use all processors
            return _sort_arr_list_into_ports_mp_main(array_list, cut_array_list, percentiles, pool)


def _sort_arr_list_into_ports_mp_main(array_list, cut_array_list, percentiles, pool,
                                      cutoffs: Optional[List[Union[float, int]]] = None):
        #For time estimation
        counter: List[float] = []
        num_loops = len(array_list)
        start_time = timeit.default_timer()

        #Mp setup
        port = partial(_create_cutoffs_arr_if_necessary_and_sort_into_ports,
                       percentiles=percentiles, cutoffs=cutoffs)

        data_tups = [(arr, cut_array_list[i]) for i, arr in enumerate(array_list)]

        results = [pool.apply_async(port, ((arr, cut_array_list[i]),), callback=counter.append) \
                        for i, arr in enumerate(array_list)]

        #Time estimation
        while len(counter) < num_loops:
            estimate_time(num_loops, len(counter), start_time)
            time.sleep(0.5)

        #Collect and output results. A timeout of 1 should be fine because
        #it should wait until completion anyway
        return [r.get(timeout=1) for r in results]


def _arr_list_to_series(array_list):
    return pd.Series(np.concatenate(array_list, axis=0))


def _sort_arr_list_into_ports_and_return_series(array_list, cut_array_list, percentiles, multiprocess,
                                                cutoffs: Optional[List[Union[float, int]]] = None):
    al = _sort_arr_list_into_ports(array_list, cut_array_list, percentiles, multiprocess, cutoffs=cutoffs)
    return _arr_list_to_series(al)


def _check_portfolio_inputs(*args, **kwargs):
    user_passed = partial(_user_passed, kwargs=kwargs)
    assert isinstance(args[0], pd.DataFrame)
    assert isinstance(args[1], str)
    user_passed_any_cutoffs = user_passed('cutoffs') or user_passed('quant_cutoffs')

    if user_passed_any_cutoffs and kwargs['ngroups'] not in [10, None, 0]: # 10 is default ngroups
        raise ValueError(f'cannot pass both cutoffs and ngroups. got {kwargs["cutoffs"]} cutoffs '
                         f'and {kwargs["ngroups"]} ngroups')

    if user_passed_any_cutoffs and kwargs['cutdf'] is not None:
        warnings.warn(f'cutdf will not be used for portfolios as cutoffs {kwargs["cutoffs"]} were passed')

    if user_passed('cutoffs') and user_passed('quant_cutoffs'):
        raise ValueError(f'cannot pass both cutoffs and quant_cutoffs. got {kwargs["cutoffs"]} cutoffs and '
                         f'{kwargs["quant_cutoffs"]} quant_cutoffs')

    if not user_passed_any_cutoffs:
        # Must be using ngroups instead of cutoffs
        assert isinstance(kwargs['ngroups'], int)


def _user_passed(key: str, kwargs: dict) -> bool:
    return key in kwargs and kwargs[key] is not None


def _assert_byvars_list(byvars):
    if byvars != None:
        if isinstance(byvars, str): byvars = [byvars]
        else:
            assert isinstance(byvars, list)
    return byvars