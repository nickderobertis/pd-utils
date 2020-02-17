import datetime
import functools
import sys
import timeit
import warnings
from itertools import chain
from multiprocessing import Pool
from typing import Optional, List, Union, Sequence

import numpy as np
import pandas as pd

from pd_utils.timer import estimate_time
from pd_utils.utils import split
from pd_utils.merge import groupby_merge


def cumulate(
    df: pd.DataFrame,
    cumvars: Union[str, List[str]],
    method: str,
    periodvar="Date",
    byvars: Optional[Union[str, List[str]]] = None,
    time: Optional[Sequence[int]] = None,
    grossify: bool = False,
    multiprocess: Union[bool, int] = True,
    replace: bool = False,
):
    """
    Cumulates a variable over time. Typically used to get cumulative returns.

    :param df:
    :param cumvars: column names to cumulate
    :param method: 'between', 'zero', or 'first'.
             If 'zero', will give returns since the original date. Note: for periods before the original date,
             this will turn positive returns negative as we are going backwards in time.
             If 'between', will give returns since the prior requested time period. Note that
             the first period is period 0.
             If 'first', will give returns since the first requested time period.
    :param periodvar:
    :param byvars: column names to use to separate by groups
    :param time: for use with method='between'. Defines which periods to calculate between.
    :param grossify: set to True to add one to all variables then subtract one at the end
    :param multiprocess: set to True to use all available processors,
                  set to False to use only one, pass an int less or equal to than number of
                  processors to use that amount of processors
    :param replace: True to return df with passed columns replaced with cumulated columns.
             False to return df with both passed columns and cumulated columns
    :return:

    :Examples:

    For example::

        For example, if our input data was for date 1/5/2006, but we had shifted dates:
             permno  date      RET  shift_date
             10516   1/5/2006  110%  1/5/2006
             10516   1/5/2006  120%  1/6/2006
             10516   1/5/2006  105%  1/7/2006
             10516   1/5/2006  130%  1/8/2006
         Then cumulate(df, 'RET', cumret='between', time=[1,3], get='RET', periodvar='shift_date') would return:
             permno  date      RET  shift_date  cumret
             10516   1/5/2006  110%  1/5/2006    110%
             10516   1/5/2006  120%  1/6/2006    120%
             10516   1/5/2006  105%  1/7/2006    126%
             10516   1/5/2006  130%  1/8/2006    130%
         Then cumulate(df, 'RET', cumret='first', periodvar='shift_date') would return:
             permno  date      RET  shift_date  cumret
             10516   1/5/2006  110%  1/5/2006    110%
             10516   1/5/2006  120%  1/6/2006    120%
             10516   1/5/2006  105%  1/7/2006    126%
             10516   1/5/2006  130%  1/8/2006    163.8%
    """

    import time as time2  # accidentally used time an an input parameter and don't want to break prior code

    if method == 'zero':
        raise NotImplementedError('method zero not implemented yet')

    # TODO [#1]: get method 'zero' of cumulate working
    #
    # Has some WIP already commited, commented out

    def log(message):
        if message != "\n":
            time = datetime.datetime.now().replace(microsecond=0)
            message = str(time) + ": " + message
        sys.stdout.write(message + "\n")
        sys.stdout.flush()

    log("Initializing cumulate.")

    df = df.copy()  # don't want to modify original dataframe

    sort_time: Optional[List[int]]
    if time:
        sort_time = sorted(time)
    else:
        sort_time = None

    if isinstance(cumvars, (str, int)):
        cumvars = [cumvars]
    assert isinstance(cumvars, list)

    assert isinstance(grossify, bool)

    if grossify:
        for col in cumvars:
            df[col] = df[col] + 1

    # For method 'zero' implementation
    # def unflip(df, cumvars):
    #     flipcols = ['cum_' + str(c) for c in cumvars] #select cumulated columns
    #     for col in flipcols:
    #         tempdf[col] = tempdf[col].shift(1) #shift all values down one row for cumvars
    #         tempdf[col] = -tempdf[col] + 2 #converts a positive return into a negative return
    #     tempdf = tempdf[1:].copy() #drop out period 0
    #     tempdf = tempdf.sort_values(periodvar) #resort to original order

    def flip(df, flip):
        flip_df = df[df["window"].isin(flip)]
        rest = df[~df["window"].isin(flip)]
        flip_df = flip_df.sort_values(byvars + [periodvar], ascending=False)
        return pd.concat([flip_df, rest], axis=0)

    def _cumulate(array_list, mp=multiprocess):
        if multiprocess:
            if isinstance(multiprocess, int):
                return _cumulate_mp(array_list, mp=mp)  # use mp # processors
            else:
                return _cumulate_mp(array_list)  # use all processors
        else:
            return _cumulate_sp(array_list)

    def _cumulate_sp(array_list):
        out_list = []
        for array in array_list:
            out_list.append(np.cumprod(array, axis=0))
        return np.concatenate(out_list, axis=0)

    def _cumulate_mp(array_list, mp=None):
        if mp:
            with Pool(mp) as pool:  # use mp # processors
                return _cumulate_mp_main(array_list, pool)
        else:
            with Pool() as pool:  # use all processors
                return _cumulate_mp_main(array_list, pool)

    def _cumulate_mp_main(array_list, pool):

        # For time estimation
        counter = []
        num_loops = len(array_list)
        start_time = timeit.default_timer()

        # Mp setup
        cum = functools.partial(np.cumprod, axis=0)
        results = [
            pool.apply_async(cum, (arr,), callback=counter.append) for arr in array_list
        ]

        # Time estimation
        while len(counter) < num_loops:
            estimate_time(num_loops, len(counter), start_time)
            time2.sleep(0.5)

        # Collect and output results. A timeout of 1 should be fine because
        # it should wait until completion anyway
        return np.concatenate([r.get(timeout=1) for r in results], axis=0)

    #####TEMPORARY CODE######
    assert method.lower() != "zero"
    #########################

    if isinstance(byvars, str):
        byvars = [byvars]

    assert method.lower() in ("zero", "between", "first")
    assert not (
        (method.lower() == "between") and (time == None)
    )  # need time for between method
    if time != None and method.lower() != "between":
        warnings.warn("Time provided but method was not between. Time will be ignored.")

    # Creates a variable containing index of window in which the observation belongs
    if method.lower() == "between":
        df = _map_windows(
            df, sort_time, method=method, periodvar=periodvar, byvars=byvars
        )
    else:
        df["__map_window__"] = 1
        df.loc[df[periodvar] == min(df[periodvar]), "__map_window__"] = 0

    ####################TEMP
    #     import pdb
    #     pdb.set_trace()
    #######################

    if not byvars:
        byvars = ["__map_window__"]
    else:
        byvars.append("__map_window__")
    assert isinstance(byvars, list)

    # need to determine when to cumulate backwards
    # check if method is zero, there only negatives and zero, and there is at least one negative in each window
    if method.lower() == "zero":
        raise NotImplementedError("need to implement method zero")
        # flip is a list of indices of windows for which the window should be flipped
        # to_flip = [j for j, window in enumerate(windows) \
        #        if all([i <= 0 for i in window]) and any([i < 0 for i in window])]
        # df = flip(df, to_flip)

    log("Creating by groups.")

    # Create by groups
    df["__key_var__"] = "__key_var__"  # container for key
    for col in [df[c].astype(str) for c in byvars]:
        df["__key_var__"] += col

    array_list = split(df, cumvars)

    #     container_array = df[cumvars].values
    full_array = _cumulate(array_list)

    new_cumvars = ["cum_" + str(c) for c in cumvars]

    cumdf = pd.DataFrame(full_array, columns=new_cumvars, dtype=np.float64)
    outdf = pd.concat([df.reset_index(drop=True), cumdf], axis=1)

    if method.lower == "zero" and flip != []:  # if we flipped some of the dataframe
        pass  # TEMPORARY

    if grossify:
        all_cumvars = cumvars + new_cumvars
        for col in all_cumvars:
            outdf[col] = outdf[col] - 1

    if replace:
        outdf.drop(cumvars, axis=1, inplace=True)
        outdf.rename(
            columns={"cum_" + str(cumvar): cumvar for cumvar in cumvars}, inplace=True
        )

    drop_cols = [col for col in outdf.columns if col.startswith("__")]

    return outdf.drop(drop_cols, axis=1)


def _map_windows(
    df, time, method="between", periodvar="Shift Date", byvars=["PERMNO", "Date"]
):
    """
    Returns the dataframe with an additional column __map_window__ containing the index of the window
    in which the observation resides. For example, if the windows are
    [[1],[2,3]], and the periods are 1/1/2000, 1/2/2000, 1/3/2000 for PERMNO 10516 with byvar
    'a', the df rows would be as follows:
         (10516, 'a', '1/1/2000', 0),
         (10516, 'a', '1/2/2000', 1),
         (10516, 'a', '1/3/2000', 1),
    """

    df = df.copy()  # don't overwrite original dataframe

    wm = functools.partial(window_mapping, time, method=method)

    df = groupby_merge(df, byvars, "transform", (wm), subset=periodvar)

    return df.rename(columns={periodvar + "_transform": "__map_window__"})


def create_windows(periods, time, method='between'):

    if method.lower() == 'first':
        windows = [[0]]
        windows += [[i for i in range(1, len(periods))]]
        return windows
    elif method.lower() == 'between':
        time = [t - time[0] for t in time] #shifts time so that first period is period 0
        windows = [[0]]
        t_bot = 0
        for i, t in enumerate(time): #pick each element of time
            if t == 0: continue #already added zero
            windows.append([i for i in range(t_bot + 1, t + 1)])
            t_bot = t
        #The last window is all the leftover periods after finishing time
        extra_windows = [[i for i, per in enumerate(periods) if i not in chain.from_iterable(windows)]]
        if extra_windows != [[]]: #don't want to add empty window
            windows += extra_windows
        return windows


def window_mapping(time, col, method='between'):
    """
    Takes a pandas series of dates as inputs, calculates windows, and returns a series of which
    windows each observation are in. To be used with groupby.transform()
    """
    windows = create_windows(col, time, method=method)
    return [n for i in range(len(col.index)) for n, window in enumerate(windows) if i in window]