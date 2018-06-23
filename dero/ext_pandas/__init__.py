import datetime
import functools
import os
import sys
import timeit
import warnings
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
from tkinter import Tk, Frame, BOTH, YES

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from numpy import nan
from pandas.tseries.offsets import CustomBusinessDay
from pandasql import PandaSQL
from pandastable import Table
from sas7bdat import SAS7BDAT

from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday, \
    USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay, \
    USLaborDay, USThanksgivingDay

from .pdutils import window_mapping, year_month_from_single_date, _check_portfolio_inputs, _assert_byvars_list, \
                     _create_cutoffs_and_sort_into_ports, _split, _sort_arr_list_into_ports_and_return_series, \
                     _to_list_if_str, _expand, _to_series_if_str, _to_name_if_series, \
                     _extract_table_names_from_sql, _get_datetime_cols, \
                     _select_long_short_ports, _portfolio_difference, split

from .regby import reg_by
from .filldata import fillna_by_groups_and_keep_one_per_group, fillna_by_groups
from ..ext_time import estimate_time


def to_csv(dataframe, path, filename, output=True, action='w', index=True):
    '''
    action='w' for overwrite, 'a' for append
    set index to False to not include index in output
    '''   
    if action == 'a':
        headers = False
    else:
        headers = True
    
    if dataframe is not None: #if dataframe exists
        filepath = os.path.join(path,filename + '.csv')
        f = open(filepath, action, encoding='utf-8')
        if output is True: print("Now saving %s" % filepath)
        try: f.write(dataframe.to_csv(encoding='utf-8', index=index, header=headers)) #could use easier dataframe.to_csv(filepath) syntax, but won't overwrite
        except: f.write(dataframe.to_csv(encoding='utf-8', index=index, header=headers).replace('\ufffd',''))
        f.close()
    else:
        print("{} does not exist.".format(dataframe)) #does nothing if dataframe doesn't exist
    
def convert_sas_date_to_pandas_date(sasdates):
    epoch = datetime.datetime(1960, 1, 1)
    
    def to_pandas(date):
        return epoch + datetime.timedelta(days=date)
    
    if isinstance(sasdates, pd.Series):
        #Below code is to reduce down to unique dates and create a mapping
        
#         unique = pd.Series(sasdates.dropna().unique()).astype(int)
#         shift = unique.apply(datetime.timedelta)
#         pd_dates = epoch + shift
        
#         for_merge = pd.concat([unique, pd_dates], axis=1)
#         for_merge.columns = [sasdates.name, 0]
        
#         orig_df = pd.DataFrame(sasdates)
#         orig_df.reset_index(inplace=True)
        
#         return for_merge.merge(orig_df, how='right', on=[sasdates.name]).sort_values('index').reset_index()[0]

        return apply_func_to_unique_and_merge(sasdates, to_pandas)
    
    
#         return pd.Series([epoch + datetime.timedelta(days=int(float(date))) if not pd.isnull(date) else nan for date in sasdates])
    else:
        return epoch + datetime.timedelta(days=sasdates)
    
def year_month_from_date(df, date='Date', yearname='Year', monthname='Month'):
    '''
    Takes a dataframe with a datetime object and creates year and month variables
    '''
    df = df.copy()
#     df[yearname] =  [date.year  for date in df[date]]
#     df[monthname] = [date.month for date in df[date]]
    df[[yearname, monthname]] = apply_func_to_unique_and_merge(df[date], year_month_from_single_date)
    
    return df

def expand_time(df, intermediate_periods=False, **kwargs):
    """
    Creates new observations in the dataset advancing the time by the int or list given. Creates a new date variable.
    See _expand_time for keyword arguments.
    
    Specify intermediate_periods=True to get periods in between given time periods, e.g.
    passing time=[12,24,36] will get periods 12, 13, 14, ..., 35, 36. 
    """
    
    if intermediate_periods:
        assert 'time' in kwargs
        time = kwargs['time']
        time = [t for t in range(min(time),max(time) + 1)]
        kwargs['time'] = time
    return _expand_time(df, **kwargs)

def _expand_time(df, datevar='Date', freq='m', time=[12, 24, 36, 48, 60], newdate='Shift Date', shiftvar='Shift'):
    '''
    Creates new observations in the dataset advancing the time by the int or list given. Creates a new date variable.
    '''
    def log(message):
        if message != '\n':
            time = datetime.datetime.now().replace(microsecond=0)
            message = str(time) + ': ' + message
        sys.stdout.write(message + '\n')
        sys.stdout.flush()
    
    log('Initializing expand_time for periods {}.'.format(time))
    
    if freq == 'd':
        log('Daily frequency, getting trading day calendar.')
        td = tradedays() #gets trading day calendar
    else:
        td = None
    
    def time_shift(shift, freq=freq, td=td):
        if freq == 'm':
            return relativedelta(months=shift)
        if freq == 'd':
            return shift * td
        if freq == 'a':
            return relativedelta(years=shift)
    
    if isinstance(time, int):
        time = [time]
    else: assert isinstance(time, list)
    
    
    log('Calculating number of rows.')
    num_rows = len(df.index)
    log('Calculating number of duplicates.')
    duplicates = len(time)
    
    #Expand number of rows
    if duplicates > 1:
        log('Duplicating observations {} times.'.format(duplicates - 1))
        df = df.append([df] * (duplicates - 1)).sort_index().reset_index(drop=True)
        log('Duplicated.')
    
    log('Creating shift variable.')
    df[shiftvar] = time * num_rows #Create a variable containing amount of time to shift
    #Now create shifted date
    log('Creating shifted date.')
    df[newdate] = [date + time_shift(int(shift)) for date, shift in zip(df[datevar],df[shiftvar])]
    log('expand_time completed.')
    
    #Cleanup and exit
    return df #.drop('Shift', axis=1)

def expand_months(df, datevar='Date', newdatevar='Daily Date', trade_days=True):
    """
    Takes a monthly dataframe and returns a daily (trade day or calendar day) dataframe. 
    For each row in the input data, duplicates that row over each trading/calendar day in the month of 
    the date in that row. Creates a new date column containing the daily date.
    
    NOTE: If the input dataset has multiple observations per month, all of these will be expanded. Therefore
    you will have one row for each trade day for each original observation. 
    
    Required inputs:
    df: pandas dataframe containing a date variable
    
    Optional inputs:
    datevar: str, name of column containing dates in the input df
    newdatevar: str, name of new column to be created containing daily dates
    tradedays: bool, True to use trading days and False to use calendar days
    """
    if trade_days:
        td = tradedays()
    else:
        td = 'D'
    
    expand = functools.partial(_expand, datevar=datevar, td=td, newdatevar=newdatevar)
    
    
    expand_all = np.vectorize(expand, otypes=[np.ndarray])
        
    days =  pd.DataFrame(np.concatenate(expand_all(df[datevar].unique()), axis=0),
                         columns=[datevar, newdatevar], dtype='datetime64')

    return df.merge(days, on=datevar, how='left')

def cumulate(df, cumvars, method, periodvar='Date',  byvars=None, time=None, grossify=False,
             multiprocess=True):
    """
    Cumulates a variable over time. Typically used to get cumulative returns. 
    
    NOTE: Method zero not yet working
    
    method = 'between', 'zero', or 'first'. 
             If 'zero', will give returns since the original date. Note: for periods before the original date, 
             this will turn positive returns negative as we are going backwards in time.
             If 'between', will give returns since the prior requested time period. Note that
             the first period is period 0.
             If 'first', will give returns since the first requested time period.
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
    byvars: string or list of column names to use to seperate by groups
    time: list of ints, for use with method='between'. Defines which periods to calculate between.
    grossify: bool, set to True to add one to all variables then subtract one at the end
    multiprocess: bool or int, set to True to use all available processors, 
                  set to False to use only one, pass an int less or equal to than number of 
                  processors to use that amount of processors 
    """
    import time as time2 #accidentally used time an an input parameter and don't want to break prior code
    
    def log(message):
        if message != '\n':
            time = datetime.datetime.now().replace(microsecond=0)
            message = str(time) + ': ' + message
        sys.stdout.write(message + '\n')
        sys.stdout.flush()
    
    log('Initializing cumulate.')
    
    df = df.copy() #don't want to modify original dataframe
    
    if time:
        sort_time = sorted(time)
    else: sort_time = None
        
    if isinstance(cumvars, (str, int)):
        cumvars = [cumvars]
    assert isinstance(cumvars, list)

    assert isinstance(grossify, bool)
    
    if grossify:
        for col in cumvars:
            df[col] = df[col] + 1
    
    def unflip(df, cumvars):
        flipcols = ['cum_' + str(c) for c in cumvars] #select cumulated columns
        for col in flipcols:
            tempdf[col] = tempdf[col].shift(1) #shift all values down one row for cumvars
            tempdf[col] = -tempdf[col] + 2 #converts a positive return into a negative return
        tempdf = tempdf[1:].copy() #drop out period 0
        tempdf = tempdf.sort_values(periodvar) #resort to original order
        
    def flip(df, flip):
        flip_df = df[df['window'].isin(flip)]
        rest = df[~df['window'].isin(flip)]
        flip_df = flip_df.sort_values(byvars + [periodvar], ascending=False)
        return pd.concat([flip_df, rest], axis=0)
    
    def _cumulate(array_list, mp=multiprocess):
        if multiprocess:
            if isinstance(multiprocess, int):
                return _cumulate_mp(array_list, mp=mp) #use mp # processors
            else:
                return _cumulate_mp(array_list) #use all processors
        else:
            return _cumulate_sp(array_list)
    
    def _cumulate_sp(array_list):
        out_list = []
        for array in array_list:
            out_list.append(np.cumprod(array, axis=0))
        return np.concatenate(out_list, axis=0)
    
    def _cumulate_mp(array_list, mp=None):
        if mp:
            with Pool(mp) as pool: #use mp # processors
                return _cumulate_mp_main(array_list, pool)
        else:
            with Pool() as pool: #use all processors
                return _cumulate_mp_main(array_list, pool)
        
    def _cumulate_mp_main(array_list, pool):
        
        #For time estimation
        counter = []
        num_loops = len(array_list)
        start_time = timeit.default_timer()
        
        #Mp setup
        cum = functools.partial(np.cumprod, axis=0)
        results = [pool.apply_async(cum, (arr,), callback=counter.append) for arr in array_list]
        
        #Time estimation
        while len(counter) < num_loops:
            estimate_time(num_loops, len(counter), start_time)
            time2.sleep(0.5)
            
        #Collect and output results. A timeout of 1 should be fine because
        #it should wait until completion anyway
        return np.concatenate([r.get(timeout=1) for r in results], axis=0)
    
    #####TEMPORARY CODE######
    assert method.lower() != 'zero'
    #########################
    
    if isinstance(byvars, str):
        byvars = [byvars]
    
    assert method.lower() in ('zero','between','first')
    assert not ((method.lower() == 'between') and (time == None)) #need time for between method
    if time != None and method.lower() != 'between':
        warnings.warn('Time provided but method was not between. Time will be ignored.')

    #Creates a variable containing index of window in which the observation belongs
    if method.lower() == 'between':
        df = _map_windows(df, sort_time, method=method, periodvar=periodvar, byvars=byvars)
    else:
        df['__map_window__'] = 1
        df.loc[df[periodvar] == min(df[periodvar]), '__map_window__'] = 0

        
    
    
    ####################TEMP
#     import pdb
#     pdb.set_trace()
    #######################
    
    
    if not byvars:  byvars = ['__map_window__']
    else: byvars.append('__map_window__')
    assert isinstance(byvars, list)
    
    #need to determine when to cumulate backwards
    #check if method is zero, there only negatives and zero, and there is at least one negative in each window
    if method.lower() == 'zero': 
        #flip is a list of indices of windows for which the window should be flipped
        flip = [j for j, window in enumerate(windows) \
               if all([i <= 0 for i in window]) and any([i < 0 for i in window])]
        df = flip(df, flip)
        

    log('Creating by groups.')

    #Create by groups
    df['__key_var__'] = '__key_var__' #container for key
    for col in [df[c].astype(str) for c in byvars]:
        df['__key_var__'] += col

    array_list = split(df, cumvars)
    
#     container_array = df[cumvars].values
    full_array = _cumulate(array_list)
    
    new_cumvars = ['cum_' + str(c) for c in cumvars]

    cumdf = pd.DataFrame(full_array, columns=new_cumvars, dtype=np.float64)
    outdf = pd.concat([df.reset_index(drop=True), cumdf], axis=1)
    
    if method.lower == 'zero' and flip != []: #if we flipped some of the dataframe
        pass #TEMPORARY
    
    
    
    if grossify:
        all_cumvars = cumvars + new_cumvars
        for col in all_cumvars:
            outdf[col] = outdf[col] - 1
    
    drop_cols = [col for col in outdf.columns if col.startswith('__')]
    
    return outdf.drop(drop_cols, axis=1)

def long_to_wide(df, groupvars, values, colindex=None, colindex_only=False):
    '''
    
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
    '''
    
    df = df.copy() #don't overwrite original
    
    #Check for duplicates
    if df.duplicated().any():
        df.drop_duplicates(inplace=True)
        warnings.warn('Found duplicate rows and deleted.')
    
    #Ensure type of groupvars is correct
    if isinstance(groupvars,str):
        groupvars = [groupvars]
    assert isinstance(groupvars, list)
    
    #Ensure type of values is correct
    if isinstance(values,str):
        values = [values]
    assert isinstance(values, list)

    if colindex_only and len(values) > 1:
        raise NotImplementedError('set colindex_only to False when passing more than one value')
    
    #Fixes for colindex
    #Use count of the row within the group for column index if not specified
    if colindex == None:
        df['__idx__'] = df.groupby(groupvars).cumcount()
        colindex = '__idx__'
    #If multiple columns are provided for colindex, combine and drop old cols
    if isinstance(colindex, list):
        df['__idx__'] = ''
        for col in colindex:
            df['__idx__'] = df['__idx__'] + '_' + df[col].astype(str)
            df.drop(col, axis=1, inplace=True)
        colindex = '__idx__'
        
    
    df['__key__'] = df[groupvars[0]].astype(str) #create key variable
    if len(groupvars) > 1: #if there are multiple groupvars, combine into one key
        for var in groupvars[1:]:
            df['__key__'] = df['__key__'] + '_' + df[var].astype(str)
    
    #Create seperate wide datasets for each value variable then merge them together
    for i, value in enumerate(values):
        if i == 0:
            combined = df.copy()
        #Create wide dataset
        raw_wide = df.pivot(index='__key__', columns=colindex, values=value)
        if not colindex_only:
            # add value name
            raw_wide.columns = [value + str(col) for col in raw_wide.columns]
        else:
            # remove _ from colindex name
            raw_wide.columns = [str(col).strip('_') for col in raw_wide.columns]
        wide = raw_wide.reset_index()

        #Merge back to original dataset
        combined = combined.merge(wide, how='left', on='__key__')
    
    return combined.drop([colindex,'__key__'] + values, axis=1).drop_duplicates().reset_index(drop=True)

def load_sas(filepath, csv=True, **read_csv_kwargs):
    """
    Loads sas sas7bdat file into a pandas DataFrame.

    :param filepath: str of location of sas7bdat file
    :param csv: when set to True, saves a csv version of the data in the same directory as the sas7bdat.
                Next time load_sas will load from the csv version rather than sas7bdat, which speeds up
                load times about 3x. If the sas7bdat file is modified more recently than the csv,
                the sas7bdat will automatically be loaded and saved to the csv again.
    :param read_csv_kwargs: kwargs to pass to pd.read_csv if csv option is True
    :return:
    """
    sas_name = os.path.basename(filepath) #e.g. dsename.sas7bdat
    folder = os.path.dirname(filepath) #location of sas file
    filename, extension = os.path.splitext(sas_name) #returns ('dsenames','.sas7bdat')
    csv_name = filename + '.csv'
    csv_path = os.path.join(folder, csv_name)
    
    if os.path.exists(csv_path) and csv:
        if os.path.getmtime(csv_path) > os.path.getmtime(filepath): #if csv was modified more recently
            #Read from csv (don't touch sas7bdat because slower loading)
            try: return pd.read_csv(csv_path, encoding='utf-8', **read_csv_kwargs)
            except UnicodeDecodeError: return pd.read_csv(csv_path, encoding='cp1252', **read_csv_kwargs)
    
    #In the case that there is no csv already, or that the sas7bdat has been modified more recently
    #Pull from SAS file
    df = SAS7BDAT(filepath).to_data_frame()
    #Write to csv file
    if csv:
        to_csv(df, folder, filename, output=False, index=False)
    return df

def averages(df, avgvars, byvars, wtvar=None, count=False, flatten=True):
    '''
    Returns equal- and value-weighted averages of variables within groups
    
    avgvars: List of strings or string of variable names to take averages of
    byvars: List of strings or string of variable names for by groups
    wtvar: String of variable to use for calculating weights in weighted average
    count: False or string of variable name, pass variable name to get count of non-missing
           of that variable within groups.
    flatten: Boolean, False to return df with multi-level index
    '''
    #Check types
    assert isinstance(df, pd.DataFrame)
    if isinstance(avgvars, str): avgvars = [avgvars]
    else:
        assert isinstance(avgvars, list)
        avgvars = avgvars.copy() # don't modify existing avgvars inplace
    assert isinstance(byvars, (str, list))
    if wtvar != None:
        assert isinstance(wtvar, str)
    
    df = df.copy()
    
    if count:
        df = groupby_merge(df, byvars, 'count', subset=count)
        avgvars += [count + '_count']
    
    g = df.groupby(byvars)
    avg_df  = g.mean()[avgvars]
    
    if wtvar == None:
        if flatten:
            return avg_df.reset_index()
        else:
            return avg_df
    
    for var in avgvars:
        colname = var + '_wavg'
        df[colname] = df[wtvar] / g[wtvar].transform('sum') * df[var]
    
    wavg_cols = [col for col in df.columns if col[-4:] == 'wavg']
    
    g = df.groupby(byvars) #recreate because we not have _wavg cols in df
    wavg_df = g.sum()[wavg_cols]
    
    outdf = pd.concat([avg_df,wavg_df], axis=1)
    
    if flatten:
        return outdf.reset_index()
    else:
        return outdf
    
def portfolio(df, groupvar, ngroups=10, byvars=None, cutdf=None, portvar='portfolio',
              multiprocess=False):
    '''
    Constructs portfolios based on percentile values of groupvar. If ngroups=10, then will form 10 portfolios,
    with portfolio 1 having the bottom 10 percentile of groupvar, and portfolio 10 having the top 10 percentile
    of groupvar.
    
    df: pandas dataframe, input data
    groupvar: string, name of variable in df to form portfolios on
    ngroups: integer, number of portfolios to form
    byvars: string, list, or None, name of variable(s) in df, finds portfolios within byvars. For example if byvars='Month',
            would take each month and form portfolios based on the percentiles of the groupvar during only that month
    cutdf: pandas dataframe or None, optionally determine percentiles using another dataset. See second note.
    portvar: string, name of portfolio variable in the output dataset
    multiprocess: bool or int, set to True to use all available processors, 
                  set to False to use only one, pass an int less or equal to than number of 
                  processors to use that amount of processors 
    
    NOTE: Resets index and drops in output data, so don't use if index is important (input data not affected)
    NOTE: If using a cutdf, MUST have the same bygroups as df. The number of observations within each bygroup
          can be different, but there MUST be a one-to-one match of bygroups, or this will NOT work correctly.
          This may require some cleaning of the cutdf first.
    NOTE: For some reason, multiprocessing seems to be slower in testing, so it is disabled by default
    '''
    #Check types
    _check_portfolio_inputs(df, groupvar, ngroups=ngroups, byvars=byvars, cutdf=cutdf, portvar=portvar)
    byvars = _assert_byvars_list(byvars)
    if cutdf != None:
        assert isinstance(cutdf, pd.DataFrame)
    else: #this is where cutdf == None, the default case
        cutdf = df
        tempcutdf = cutdf.copy()
    
    pct_per_group = 100/ngroups
    percentiles = [i*pct_per_group for i in range(ngroups)] #percentile values, e.g. 0, 10, 20, 30... 100
    percentiles += [100]
    
#     pct_per_group = int(100/ngroups)
#     percentiles = [i for i in range(0, 100 + pct_per_group, pct_per_group)] #percentile values, e.g. 0, 10, 20, 30... 100
    
    #Create new functions with common arguments added
    create_cutoffs_and_sort_into_ports = functools.partial(_create_cutoffs_and_sort_into_ports, 
                                       groupvar=groupvar, portvar=portvar, percentiles=percentiles)
    split = functools.partial(_split, keepvars=[groupvar], force_numeric=True)
    sort_arr_list_into_ports_and_return_series = functools.partial(_sort_arr_list_into_ports_and_return_series,
                                                         percentiles=percentiles,
                                                         multiprocess=multiprocess)
    
    tempdf = df.copy()
    
    #If there are no byvars, just complete portfolio sort
    if byvars == None: return create_cutoffs_and_sort_into_ports(tempdf, cutdf)
    
    #The below rename is incase there is already a variable named index in the data
    #The rename will just not do anything if there's not
    tempdf = tempdf.reset_index(drop=True).rename(
        columns={'index':'__temp_index__'}).reset_index() #get a variable 'index' containing obs count
    
    #Also replace index in byvars if there
    temp_byvars = [b if b != 'index' else '__temp_index__' for b in byvars]
    all_byvars = [temp_byvars, byvars] #list of lists
    
    #else, deal with byvars
    #First create a key variable based on all the byvars
    for i, this_df in enumerate([tempdf, tempcutdf]):
        this_df['__key_var__'] = 'key' #container for key
        for col in [this_df[c].astype(str) for c in all_byvars[i]]:
            this_df['__key_var__'] += col
        this_df.sort_values('__key_var__', inplace=True)
    
    #Now split into list of arrays and process
    array_list = split(tempdf)
    cut_array_list = split(tempcutdf)

    tempdf = tempdf.reset_index(drop=True) #need to reset index again for adding new column
    tempdf[portvar] = sort_arr_list_into_ports_and_return_series(array_list, cut_array_list)
    return tempdf.sort_values('index').drop(['__key_var__','index'], axis=1).rename(
                columns={'__temp_index__':'index'}).reset_index(drop=True)

    
def portfolio_averages(df, groupvar, avgvars, ngroups=10, byvars=None, cutdf=None, wtvar=None,
                       count=False, portvar='portfolio', avgonly=False):
    '''
    Creates portfolios and calculates equal- and value-weighted averages of variables within portfolios. If ngroups=10,
    then will form 10 portfolios, with portfolio 1 having the bottom 10 percentile of groupvar, and portfolio 10 having 
    the top 10 percentile of groupvar.
    
    df: pandas dataframe, input data
    groupvar: string, name of variable in df to form portfolios on
    avgvars: string or list, variables to be averaged
    ngroups: integer, number of portfolios to form
    byvars: string, list, or None, name of variable(s) in df, finds portfolios within byvars. For example if byvars='Month',
            would take each month and form portfolios based on the percentiles of the groupvar during only that month
    cutdf: pandas dataframe or None, optionally determine percentiles using another dataset
    wtvar: string, name of variable in df to use for weighting in weighted average
    count: False or string of variable name, pass variable name to get count of non-missing
           of that variable within groups.
    portvar: string, name of portfolio variable in the output dataset
    avgonly: boolean, True to return only averages, False to return (averages, individual observations with portfolios)
    
    NOTE: Resets index and drops in output data, so don't use if index is important (input data not affected)
    '''
    ports = portfolio(df, groupvar, ngroups=ngroups, byvars=byvars, cutdf=cutdf, portvar=portvar)
    if byvars:
        assert isinstance(byvars, (str, list))
        if isinstance(byvars, str): byvars = [byvars]
        by = [portvar] + byvars
        avgs = averages(ports, avgvars, byvars=by, wtvar=wtvar, count=count)
    else:
        avgs = averages(ports, avgvars, byvars=portvar, wtvar=wtvar, count=count)
    
    if avgonly:
        return avgs
    else:
        return avgs, ports


def factor_reg_by(df, groupvar, fac=4, retvar='RET', mp=False, stderr=False):
    """
    Takes a dataframe with RET, mktrf, smb, hml, and umd, and produces abnormal returns by groups.
    
    Required inputs:
    df: pandas datafram containing mktrf, smb, hml, umd, (or what's required for chosen model)
        and a return variable
    groupvar: str or list of strs, column names of columns on which to form by groups
    fac: int (1, 3, 4), factor model to run
    retvar: str, name of column containing returns. risk free rate will be subtracted from this column
    stderr: bool, True to include standard errors of coefficients

    Optional Inputs:
    mp: False to use single processor, True to use all processors, int to use # processors

    """
    assert fac in (1, 3, 5)
    factors = ['mktrf']
    if fac >= 3:
        factors += ['smb','hml']
    if fac == 5:
        factors += ['rmw','cma']

    # Create returns in excess of risk free rate
    excess_var ='_' + retvar + '_minus_rf'
    df[excess_var] = df[retvar] - df['rf']
        
    outdf = reg_by(df, excess_var, factors, groupvar, merge=True, mp=mp, stderr=stderr)
    outdf['AB' + retvar] = outdf[retvar] - sum([outdf[fac] * outdf['coef_' + fac].astype(float) for fac in factors]) #create abnormal returns

    # Cleanup excess returns
    outdf.drop(excess_var, axis=1, inplace=True)

    return outdf

def state_abbrev(df, col, toabbrev=False):
    df = df.copy()
    states_to_abbrev = {
    'Alabama': 'AL', 
    'Montana': 'MT',
    'Alaska': 'AK', 
    'Nebraska': 'NE',
    'Arizona': 'AZ', 
    'Nevada': 'NV',
    'Arkansas': 'AR', 
    'New Hampshire': 'NH',
    'California': 'CA', 
    'New Jersey': 'NJ',
    'Colorado': 'CO', 
    'New Mexico': 'NM',
    'Connecticut': 'CT', 
    'New York': 'NY',
    'Delaware': 'DE', 
    'North Carolina': 'NC',
    'Florida': 'FL', 
    'North Dakota': 'ND',
    'Georgia': 'GA', 
    'Ohio': 'OH',
    'Hawaii': 'HI', 
    'Oklahoma': 'OK',
    'Idaho': 'ID', 
    'Oregon': 'OR',
    'Illinois': 'IL', 
    'Pennsylvania': 'PA',
    'Indiana': 'IN', 
    'Rhode Island': 'RI',
    'Iowa': 'IA', 
    'South Carolina': 'SC',
    'Kansas': 'KS', 
    'South Dakota': 'SD',
    'Kentucky': 'KY', 
    'Tennessee': 'TN',
    'Louisiana': 'LA', 
    'Texas': 'TX',
    'Maine': 'ME', 
    'Utah': 'UT',
    'Maryland': 'MD', 
    'Vermont': 'VT',
    'Massachusetts': 'MA', 
    'Virginia': 'VA',
    'Michigan': 'MI', 
    'Washington': 'WA',
    'Minnesota': 'MN', 
    'West Virginia': 'WV',
    'Mississippi': 'MS', 
    'Wisconsin': 'WI',
    'Missouri': 'MO', 
    'Wyoming': 'WY', }
    if toabbrev:
        df[col] = df[col].replace(states_to_abbrev)
    else:
        abbrev_to_states = dict ( (v,k) for k, v in states_to_abbrev.items() )
        df[col] = df[col].replace(abbrev_to_states)
    
    return df

class USTradingCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday('NewYearsDay', month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday('USIndependenceDay', month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday('Christmas', month=12, day=25, observance=nearest_workday)
    ]
        
def tradedays():
    """
    Used for constructing a range of dates with pandas date_range function.

    :Example:

    >>>import pandas as pd
    >>>pd.date_range(
    >>>    start='1/1/2000',
    >>>    end='1/31/2000',
    >>>    freq=dero.pandas.tradedays()
    >>>)
    pd.DatetimeIndex(['2000-01-03', '2000-01-04', '2000-01-05', '2000-01-06',
               '2000-01-07', '2000-01-10', '2000-01-11', '2000-01-12',
               '2000-01-13', '2000-01-14', '2000-01-18', '2000-01-19',
               '2000-01-20', '2000-01-21', '2000-01-24', '2000-01-25',
               '2000-01-26', '2000-01-27', '2000-01-28', '2000-01-31'],
              dtype='datetime64[ns]', freq='C')

    """
    trading_calendar = USTradingCalendar()
    return CustomBusinessDay(holidays=trading_calendar.holidays())

def select_rows_by_condition_on_columns(df, cols, condition='== 1', logic='or'):
    """
    Selects rows of a pandas dataframe by evaluating a condition on a subset of the dataframe's columns.
    
    df: pandas dataframe
    cols: list of column names, the subset of columns on which to evaluate conditions
    condition: string, needs to contain comparison operator and right hand side of comparison. For example,
               '== 1' checks for each row that the value of each column is equal to one.
    logic: 'or' or 'and'. With 'or', only one of the columns in cols need to match the condition for the row to be kept.
            With 'and', all of the columns in cols need to match the condition.
    """
    #First eliminate spaces in columns, this method will not work with spaces
    new_cols = [col.replace(' ','_').replace('.','_') for col in cols]
    df.rename(columns={col:new_col for col, new_col in zip(cols, new_cols)}, inplace=True)
    
    #Now create a string to query the dataframe with
    logic_spaces = ' ' + logic + ' '
    query_str = logic_spaces.join([str(col) + condition for col in new_cols]) #'col1 == 1, col2 == 1', etc.
    
    #Query dataframe
    outdf = df.query(query_str).copy()
    
    #Rename columns back to original
    outdf.rename(columns={new_col:col for col, new_col in zip(cols, new_cols)}, inplace=True)
    
    return outdf

def show_df(df):
    pool = ThreadPool(1)
    pool.apply_async(_show_df, args=[df])
    
def _show_df(df):
    root = Tk()
    frame = Frame(root)
    frame.pack(fill=BOTH, expand=YES)
    pt = Table(parent=frame, dataframe=df)
    pt.show()
    pt.queryBar()
    root.mainloop()
    
def groupby_merge(df, byvars, func_str, *func_args, subset='all', replace=False):
    """
    Creates a pandas groupby object, applies the aggregation function in func_str, and merges back the 
    aggregated data to the original dataframe.
    
    Required Inputs:
    df: Pandas DataFrame
    byvars: str or list, column names which uniquely identify groups
    func_str: str, name of groupby aggregation function such as 'min', 'max', 'sum', 'count', etc.
    
    Optional Input:
    subset: str or list, column names for which to apply aggregation functions
    func_args: tuple, arguments to pass to func
    replace: bool, True to replace original columns in the data with aggregated/transformed columns
    
    Usage:
    df = groupby_merge(df, ['PERMNO','byvar'], 'max', subset='RET')
    """
    
    #Convert byvars to list if neceessary
    if isinstance(byvars, str):
        byvars = [byvars]
    
    #Store all variables except byvar in subset if subset is 'all'
    if subset == 'all':
        subset = [col for col in df.columns if col not in byvars]
        
    #Convert subset to list if necessary
    if isinstance(subset, str):
        subset = [subset]

    # Groupby expects to receive a string if there is a single variable
    if len(subset) == 1:
        groupby_subset = subset[0]
    else:
        groupby_subset = subset
    
    if func_str == 'transform':
        #transform works very differently from other aggregation functions
        
        #First we need to deal with nans in the by variables. If there are any nans, transform will error out
        #Therefore we must fill the nans in the by variables beforehand and replace afterwards
        df[byvars] = df[byvars].fillna(value='__tempnan__')
        
        #Now we must deal with nans in the subset variables. If there are any nans, tranform will error out
        #because it tries to ignore the nan. Therefore we must remove these rows from the dataframe,
        #transform, then add those rows back.
        any_nan_subset_mask = pd.Series([all(i) for i in \
                                        (zip(*[~pd.isnull(df[col]) for col in subset]))],
                                        index=df.index)
        no_nans = df[any_nan_subset_mask]
        
        grouped = no_nans.groupby(byvars)
        func = getattr(grouped, func_str) #pull method of groupby class with same name as func_str
        grouped = func(*func_args)[groupby_subset] #apply the class method and select subset columns
        if isinstance(grouped, pd.DataFrame):
            grouped.columns = [col + '_' + func_str for col in grouped.columns] #rename transformed columns
        elif isinstance(grouped, pd.Series):
            grouped.name = str(grouped.name) + '_' + func_str
        
        df.replace('__tempnan__', nan, inplace=True) #fill nan back into dataframe
        
        #Put nan rows back
        grouped = grouped.reindex(df.index)
        
        full = pd.concat([df, grouped], axis=1)
        
    else: #.min(), .max(), etc.
        
        
        
#         grouped = df.groupby(byvars, as_index=False)[byvars + subset]
#         func = getattr(grouped, func_str) #pull method of groupby class with same name as func_str
#         grouped = func(*func_args) #apply the class method


        grouped = df.groupby(byvars)[groupby_subset]
        func = getattr(grouped, func_str) #pull method of groupby class with same name as func_str
        grouped = func(*func_args) #apply the class method
        grouped = grouped.reset_index()
        
        
        #Merge and output
        full = df.merge(grouped, how='left', on=byvars, suffixes=['','_' + func_str])
    
    if replace:
        _replace_with_transformed(full, func_str)
    
    return full
    
def _replace_with_transformed(df, func_str='transform'):
    transform_cols = [col for col in df.columns if col.endswith('_' + func_str)]
    orig_names = [col[:col.find('_' + func_str)] for col in transform_cols]
    df.drop(orig_names, axis=1, inplace=True)
    df.rename(columns={old: new for old, new in zip(transform_cols, orig_names)}, inplace=True)
    
def groupby_index(df, byvars, sortvars=None, ascending=True):
    """
    Returns a dataframe which is a copy of the old one with an additional column containing an index
    by groups. Each time the bygroup changes, the index restarts at 0.
    
    Required inputs:
    df: pandas DataFrame
    byvars: str or list of column names containing group identifiers
    
    Optional inputs:
    sortvars: str or list of column names to sort by within by groups
    ascending: bool, direction of sort
    """
    
    #Convert sortvars to list if necessary
    if isinstance(sortvars, str):
        sortvars = [sortvars]
    if sortvars == None: sortvars = []
    
    df = df.copy() #don't modify the original dataframe
    df.sort_values(byvars + sortvars, inplace=True, ascending=ascending)
    df['__temp_cons__'] = 1
    df = groupby_merge(df, byvars, 'transform', (lambda x: [i for i in range(len(x))]), subset=['__temp_cons__'])
    df.drop('__temp_cons__', axis=1, inplace=True)
    return df.rename(columns={'__temp_cons___transform': 'group_index'})

def to_copy_paste(df, index=False, column_names=True):
    """
    Takes a dataframe and prints all of its data in such a format that it can be copy-pasted to create
    a new dataframe from the pandas.DataFrame() constructor.
    
    Required inputs:
    df: pandas dataframe
    
    Optional inputs:
    index: bool, True to include index
    column_names: bool, False to exclude column names
    """
    print('pd.DataFrame(data = [')
    for tup in df.iterrows():        
        data = tup[1].values
        print(str(tuple(data)) + ',')
    last_line = ']'
    if column_names:
        last_line += ', columns = {}'.format([i for i in df.columns]) #list comp to remove Index() around cols
    if index:
        last_line += ',\n index = {}'.format([i for i in df.index]) #list comp to remove Index() around index
    last_line += ')' #end command
    print(last_line)
    
def _join_col_strings(*args):
    strs = [str(arg) for arg in args]
    return '_'.join(strs)

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
    
    #Check inputs
    assert any([bot, top]) #must winsorize something
    if isinstance(pct, float):
        bot_pct = pct
        top_pct = 1 - pct
    elif isinstance(pct, list):
        bot_pct = pct[0]
        top_pct = 1 - pct[1]
    else:
        raise ValueError('pct must be float or a list of two floats')
        
    def temp_winsor(col):
        return _winsorize(col, top_pct, bot_pct, top=top, bot=bot)

    #Save column order
    cols = df.columns
    
    #Get a dataframe of data to be winsorized, and a dataframe of the other columns
    to_winsor, rest = _select_numeric_or_subset(df, subset, extra_include=byvars)

    #Now winsorize
    if byvars: #use groupby to process groups individually
        to_winsor = groupby_merge(to_winsor, byvars, 'transform', (temp_winsor), replace=True)
    else: #do entire df, one column at a time
        to_winsor.apply(temp_winsor, axis=0)
    
    return pd.concat([to_winsor,rest], axis=1)[cols]


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
        subset    = to_winsor.columns
        rest      = df.select_dtypes(exclude=[np.number, np.int64]).copy()
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

def apply_func_to_unique_and_merge(series, func):
    """
    Many Pandas functions can be slow because they're doing repeated work. This function reduces
    the given series down to unique values, applies the function, then expands back up to the
    original shape of the data. Returns a series.
    
    Required inputs:
    seres: pd.Series
    func: function to be applied to the series.
    
    :Usage:

    >>>import functools
    >>>to_datetime = functools.partial(pd.to_datetime, format='%Y%m')
    >>>apply_func_to_unique_and_merge(df['MONTH'], to_datetime)
    """

    unique = pd.Series(series.dropna().unique())
    new = unique.apply(func)

    for_merge = pd.concat([unique, new], axis=1)
    num_cols = [i for i in range(len(for_merge.columns) - 1)] #names of new columns
    for_merge.columns = [series.name] + num_cols

    orig_df = pd.DataFrame(series)
    orig_df.reset_index(inplace=True)

    return for_merge.merge(orig_df, how='right', on=[series.name]).sort_values('index').reset_index().loc[:,num_cols]


def _map_windows(df, time, method='between', periodvar='Shift Date', byvars=['PERMNO','Date']):
    """
    Returns the dataframe with an additional column __map_window__ containing the index of the window 
    in which the observation resides. For example, if the windows are
    [[1],[2,3]], and the periods are 1/1/2000, 1/2/2000, 1/3/2000 for PERMNO 10516 with byvar
    'a', the df rows would be as follows:
         (10516, 'a', '1/1/2000', 0),
         (10516, 'a', '1/2/2000', 1),
         (10516, 'a', '1/3/2000', 1),
    """

    df = df.copy() #don't overwrite original dataframe
    
    wm = functools.partial(window_mapping, time, method=method)

    df = groupby_merge(df, byvars, 'transform', (wm), subset=periodvar)

    return df.rename(columns={periodvar + '_transform': '__map_window__'})

def left_merge_latest(df, df2, on, left_datevar='Date', right_datevar='Date',
                      limit_years=False, backend='pandas'):
    """
    Left merges df2 to df using on, but grabbing the most recent observation (right_datevar will be
    the soonest earlier than left_datevar). Useful for situations where data needs to be merged with
    mismatched dates, and just the most recent data available is needed. 
    
    Required inputs:
    df: Pandas dataframe containing source data (all rows will be kept), must have on variables
        and left_datevar
    df2: Pandas dataframe containing data to be merged (only the most recent rows before source
        data will be kept)
    on: str or list of strs, names of columns on which to match, excluding date
    
    Optional inputs:
    left_datevar: str, name of date variable on which to merge in df
    right_datevar: str, name of date variable on which to merge in df2
    limit_years: False or int, only applicable for backend='sql'. 
    backend: str, 'pandas' or 'sql'. Specify the underlying machinery used to perform the merge.
             'pandas' means native pandas, while 'sql' uses pandasql. Try 'sql' if you run
             out of memory.
    
    """
    if isinstance(on, str):
        on = [on]
        
    if backend.lower() in ('pandas','pd'):
        return _left_merge_latest_pandas(df, df2, on, left_datevar=left_datevar, right_datevar=right_datevar)
    elif backend.lower() in ('sql','pandasql'):
        return _left_merge_latest_sql(df, df2, on, left_datevar=left_datevar, right_datevar=right_datevar)
    else:
        raise ValueError("select backend='pandas' or backend='sql'.")
        
    
def _left_merge_latest_pandas(df, df2, on, left_datevar='Date', right_datevar='Date'):
    many = df.loc[:,on + [left_datevar]].merge(df2, on=on, how='left')
    
    rename = False
    #if they are named the same, pandas will automatically add _x and _y to names
    if left_datevar == right_datevar: 
        rename = True #will need to rename the _x datevar for the last step
        orig_left_datevar = left_datevar
        left_datevar += '_x'
        right_datevar += '_y'
    
    lt = many.loc[many[left_datevar] >= many[right_datevar]] #left with datadates less than date

    #find rows within groups which have the maximum right_datevar (soonest before left_datevar)
    data_rows = lt.groupby(on + [left_datevar], as_index=False)[right_datevar].max() \
        .merge(lt, on=on + [left_datevar, right_datevar], how='left')
        
    if rename: #remove the _x for final merge
        data_rows.rename(columns={left_datevar: orig_left_datevar}, inplace=True)
        return df.merge(data_rows, on=on + [orig_left_datevar], how='left')
    
    #if no renaming is required, just merge and exit
    return df.merge(data_rows, on=on + [left_datevar], how='left')

def _left_merge_latest_sql(df, df2, on, left_datevar='Date', right_datevar='Date'):
    
    if left_datevar == right_datevar:
        df2 = df2.copy()
        df2.rename(columns={right_datevar: right_datevar + '_y'}, inplace=True)
        right_datevar += '_y'
    
    on_str = ' and \n    '.join(['a.{0} = b.{0}'.format(i) for i in on])
    groupby_str = ', '.join(on)
    a_cols = ', '.join(['a.' + col for col in on + [left_datevar]])
    b_cols = ', '.join(['b.' + col for col in df2.columns if col not in on])
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
    """.format(on_str, left_datevar, right_datevar, groupby_str, b_cols, a_cols)
    
    return df.merge(sql([df, df2], query), on=on + [left_datevar], how='left')
    

def var_change_by_groups(df, var, byvars, datevar='Date', numlags=1):
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
    var, byvars, datevar = [_to_list_if_str(v) for v in [var, byvars, datevar]] #convert to lists
    short_df = df.loc[~pd.isnull(df[byvars]).any(axis=1), var + byvars + datevar].drop_duplicates()
    for v in var:
        short_df[v + '_lag'] = short_df.groupby(byvars)[v].shift(numlags)
        short_df[v + '_change'] = short_df[v] - short_df[v + '_lag']
    dropvars = [v for v in var] + [v  + '_lag' for v in var]
    short_df = short_df.drop(dropvars, axis=1)
    return df.merge(short_df, on=datevar + byvars, how='left')

def fill_excluded_rows(df, byvars, fillvars=None, **fillna_kwargs):
    """
    Takes a dataframe which does not contain all possible combinations of byvars as rows. Creates
    those rows if fillna_kwargs are passed, calls fillna using fillna_kwargs for fillvars.
    
    For example, df:
                 date     id  var
        0  2003-06-09 42223C    1
        1  2003-06-10 09255G    2
    with fillna_for_excluded_rows(df, byvars=['date','id'], fillvars='var', value=0) becomes:
                  date     id  var
        0  2003-06-09 42223C    1
        1  2003-06-10 42223C    0
        2  2003-06-09 09255G    0
        3  2003-06-10 09255G    2
        
    Required options:
    df: pandas dataframe
    byvars: variables on which dataset should be expanded to product. Can pass a str, list of 
            strs, or a list of pd.Series.
    
    Optional options:
    fillvars: variables to apply fillna to
    fillna_kwargs: See pandas.DataFrame.fillna for kwargs, value=0 is common
    
    
    """
    byvars, fillvars = [_to_list_if_str(v) for v in [byvars, fillvars]] #convert to lists
    
    
#     multiindex = [df[i].dropna().unique() for i in byvars]
    multiindex = [_to_series_if_str(df, i).dropna().unique() for i in byvars]
    byvars = [_to_name_if_series(i) for i in byvars] #get name of any series


    all_df = pd.DataFrame(index=pd.MultiIndex.from_product(multiindex)).reset_index()
    all_df.columns = byvars
    merged = all_df.merge(df, how='left', on=byvars)
    
    if fillna_kwargs:
        fillna_kwargs.update({'inplace':False})
        merged[fillvars] = merged[fillvars].fillna(**fillna_kwargs)
    return merged

def sql(df_list, query):
    """
    Convenience function for running a pandasql query. Keeps track of which variables are of
    datetime type, and converts them back after running the sql query.
    
    NOTE: Ensure that dfs are passed in the order that they are used in the query.
    """
    #Pandasql looks up tables by names given in query. Here we are passed a list of dfs without names.
    #Therefore we need to extract the names of the tables from the query, then assign 
    #those names to the dfs in df_list in the locals dictionary.
    table_names = _extract_table_names_from_sql(query)
    for i, name in enumerate(table_names):
        locals().update({name: df_list[i]})
    
    #Get date variable column names
    datevars = []
#     othervars = []
    for d in df_list:
        datevars += _get_datetime_cols(d)
#         othervars += [col for col in d.columns if col not in datevars]
    datevars = list(set(datevars)) #remove duplicates
#     othervars = list(set(othervars))
    
    merged = PandaSQL()(query)
    
    #Convert back to datetime
    for date in [d for d in datevars if d in merged.columns]:
        merged[date] = pd.to_datetime(merged[date])
    return merged

def long_short_portfolio(df, portvar, byvars=None, retvars=None, top_minus_bot=True):
    """
    Takes a df with a column of numbered portfolios and creates a new
    portfolio which is long the top portfolio and short the bottom portfolio. 
    Returns a df of long-short portfolio
    
    Required inputs:
    df: pandas dataframe containing a column with portfolio numbers
    portvar: str, name of column containing portfolios
    
    Optional inputs:
    byvars: str or list of strs of column names containing groups for portfolios.
            Calculates long-short within these groups. These should be the same groups
            in which portfolios were formed.
    retvars: str or list of strs of variables to return in the long-short dataset. 
            By default, will use all numeric variables in the df.
    top_minus_bot: boolean, True to be long the top portfolio, short the bottom portfolio.
                   False to be long the bottom portfolio, short the top portfolio.
    """
    long, short = _select_long_short_ports(df, portvar, top_minus_bot=top_minus_bot)
    return _portfolio_difference(df, portvar, long, short, byvars=byvars, retvars=retvars)