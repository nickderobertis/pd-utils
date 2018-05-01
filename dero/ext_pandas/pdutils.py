
import pandas as pd
import numpy as np
from itertools import chain
from collections import OrderedDict
from functools import partial
from multiprocessing import Pool
from dateutil.relativedelta import relativedelta
import timeit
import time, datetime
import re

from dero.ext_time import estimate_time
from dero.core import OrderedSet

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


def year_month_from_single_date(date):
    d = OrderedDict()
    d.update({'Year': date.year})
    d.update({'Month': date.month})
    return pd.Series(d)

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


############Portfolio Utilities###############
def _check_portfolio_inputs(*args, **kwargs):
    assert isinstance(args[0], pd.DataFrame)
    assert isinstance(args[1], str)
    assert isinstance(kwargs['ngroups'], int)
    
def _assert_byvars_list(byvars):
    if byvars != None:
        if isinstance(byvars, str): byvars = [byvars]
        else:
            assert isinstance(byvars, list)
    return byvars

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


def _sort_arr_list_into_ports(array_list, cut_array_list, percentiles, multiprocess):
    if multiprocess:
        if isinstance(multiprocess, int):
            return _sort_arr_list_into_ports_mp(array_list, cut_array_list, percentiles, mp=multiprocess)
        else:
            return _sort_arr_list_into_ports_mp(array_list, cut_array_list, percentiles)
    else:
        return _sort_arr_list_into_ports_sp(array_list, cut_array_list, percentiles)


def _create_cutoffs_arr_and_sort_into_ports(data_tup, percentiles):
    arr, cutarr = data_tup
    cutoffs = _create_cutoffs_arr(cutarr, percentiles)
    if cutoffs:
        return _sort_arr_into_ports(arr, cutoffs)
    else:
        return [0 for elem in arr]
    
def _sort_arr_list_into_ports_sp(array_list, cut_array_list, percentiles):
    outlist = []
    for i, arr in enumerate(array_list):
        result = _create_cutoffs_arr_and_sort_into_ports((arr, cut_array_list[i]),
                                                        percentiles)
        outlist.append(result)
    return outlist

def _sort_arr_list_into_ports_mp(array_list, cut_array_list, percentiles, mp=None):
    if mp:
        with Pool(mp) as pool: #use mp # processors
            return _sort_arr_list_into_ports_mp_main(array_list, cut_array_list, percentiles, pool)
    else:
        with Pool() as pool: #use all processors
            return _sort_arr_list_into_ports_mp_main(array_list, cut_array_list, percentiles, pool)

    
def _sort_arr_list_into_ports_mp_main(array_list, cut_array_list, percentiles, pool):
        #For time estimation
        counter = []
        num_loops = len(array_list)
        start_time = timeit.default_timer()
        
        #Mp setup
        port = partial(_create_cutoffs_arr_and_sort_into_ports,
                                 percentiles=percentiles)
        
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

def _sort_arr_list_into_ports_and_return_series(array_list, cut_array_list, percentiles, multiprocess):
    al = _sort_arr_list_into_ports(array_list, cut_array_list, percentiles, multiprocess)
    return _arr_list_to_series(al)

#############End portfolio utilities###########################################################################

def _expand(monthly_date, datevar, td, newdatevar):
   
    t = time.gmtime(monthly_date/1000000000) #date coming in as integer, need to parse
    t = datetime.date(t.tm_year, t.tm_mon, t.tm_mday) #better output than gmtime
    
    beginning = datetime.date(t.year, t.month, 1) #beginning of month of date
    end = beginning + relativedelta(months=1, days=-1) #last day of month
    days = pd.date_range(start=beginning, end=end, freq=td) #trade days within month
    days.name = newdatevar
    result =  np.array([(t, i) for i in days])
    return result


def _extract_table_names_from_sql(query):
    """ Extract table names from an SQL query. """
    # a good old fashioned regex. turns out this worked better than actually parsing the code
    tables_blocks = re.findall(r'(?:FROM|JOIN)\s+(\w+(?:\s*,\s*\w+)*)', query, re.IGNORECASE)
    tables = [tbl
              for block in tables_blocks
              for tbl in re.findall(r'\w+', block)]
    return OrderedSet(tables)

def _get_datetime_cols(df):
    """
    Returns a list of column names of df for which the dtype starts with datetime
    """
    dtypes = df.dtypes
    return dtypes.loc[dtypes.apply(lambda x: str(x).startswith('datetime'))].index.tolist()

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