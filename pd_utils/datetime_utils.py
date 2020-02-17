import datetime
import functools
import sys
import time
from collections import OrderedDict
from typing import Optional, Union, List

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday, USMartinLutherKingJr, \
    USPresidentsDay, GoodFriday, USMemorialDay, USLaborDay, USThanksgivingDay
from pandas.tseries.offsets import CustomBusinessDay

from pd_utils.merge import apply_func_to_unique_and_merge


def convert_sas_date_to_pandas_date(sasdates: Union[pd.Series, int]) -> Union[pd.Series, pd.Timestamp]:
    """
    Converts a date or Series of dates loaded from a SAS SAS7BDAT file to a pandas date type.

    :param sasdates: SAS7BDAT-loaded date(s) to convert
    :return:
    """
    epoch = datetime.datetime(1960, 1, 1)

    def to_pandas(date):
        if pd.isnull(date):
            return pd.NaT
        return epoch + datetime.timedelta(days=date)

    if isinstance(sasdates, pd.Series):
        return apply_func_to_unique_and_merge(sasdates, to_pandas)
    else:
        return to_pandas(sasdates)


def year_month_from_date(df: pd.DataFrame, date: str = "Date", yearname: str = "Year", monthname: str = "Month"):
    """
    Takes a dataframe with a datetime object and creates year and month variables

    :param df:
    :param date: name of date column
    :param yearname: name of year column to be created
    :param monthname: name of month column to be created
    :return:
    """
    df = df.copy()
    df[yearname] =  [date.year  for date in df[date]]
    df[monthname] = [date.month for date in df[date]]
    df[[yearname, monthname]] = apply_func_to_unique_and_merge(df[date], year_month_from_single_date)

    return df


def expand_time(
    df: pd.DataFrame,
    intermediate_periods: bool = False,
    datevar: str = "Date",
    freq: str = "m",
    time: List[int] = [12, 24, 36, 48, 60],
    newdate: str = "Shift Date",
    shiftvar: str = "Shift",
    custom_business_day: Optional[CustomBusinessDay] = None,
):
    """
    Creates new observations in the dataset advancing the time by the int or list given. Creates a new date variable.

    :param df:
    :param intermediate_periods: Specify intermediate_periods=True to get periods in between given time periods, e.g.
        passing time=[12,24,36] will get periods 12, 13, 14, ..., 35, 36.
    :param datevar: column name of date variable
    :param freq: 'd' for daily, 'm' for monthly, 'a' for annual
    :param time: number of periods to advance by
    :param newdate: name of new date in the output data
    :param shiftvar: name of variable which specifies how much the time has been shifted
    :param custom_business_day: Only used for daily frequency. Defaults to using
        trading days based on US market holiday calendar. Can pass custom business days for other calendars
    :return:
    """

    if intermediate_periods:
        time = [t for t in range(min(time), max(time) + 1)]
    return _expand_time(
        df,
        datevar=datevar,
        freq=freq,
        time=time,
        newdate=newdate,
        shiftvar=shiftvar,
        custom_business_day=custom_business_day
    )


def _expand_time(
    df: pd.DataFrame,
    datevar: str = "Date",
    freq: str = "m",
    time: List[int] = [12, 24, 36, 48, 60],
    newdate: str = "Shift Date",
    shiftvar: str = "Shift",
    custom_business_day: Optional[CustomBusinessDay] = None,
):
    """
    Creates new observations in the dataset advancing the time by the int or list given. Creates a new date variable.

    :param df:
    :param datevar: column name of date variable
    :param freq: 'd' for daily, 'm' for monthly, 'a' for annual
    :param time: number of periods to advance by
    :param newdate: name of new date in the output data
    :param shiftvar: name of variable which specifies how much the time has been shifted
    :param custom_business_day: Only used for daily frequency. Defaults to using
        trading days based on US market holiday calendar. Can pass custom business days for other calendars
    :return:
    """

    def log(message):
        if message != "\n":
            time = datetime.datetime.now().replace(microsecond=0)
            message = str(time) + ": " + message
        sys.stdout.write(message + "\n")
        sys.stdout.flush()

    log("Initializing expand_time for periods {}.".format(time))

    if freq == "d":

        if custom_business_day is None:
            log("Daily frequency, getting trading day calendar.")
            td = tradedays()  # gets trading day calendar
        else:
            log("Daily frequency, using passed business day calendar.")
            td = custom_business_day
    else:
        td = None

    def time_shift(shift, freq=freq, td=td):
        if freq == "m":
            return relativedelta(months=shift)
        if freq == "d":
            return shift * td
        if freq == "a":
            return relativedelta(years=shift)

    if isinstance(time, int):
        time = [time]
    else:
        assert isinstance(time, list)

    log("Calculating number of rows.")
    num_rows = len(df.index)
    log("Calculating number of duplicates.")
    duplicates = len(time)

    # Expand number of rows
    if duplicates > 1:
        log("Duplicating observations {} times.".format(duplicates - 1))
        df = df.append([df] * (duplicates - 1)).sort_index().reset_index(drop=True)
        log("Duplicated.")

    log("Creating shift variable.")
    df[shiftvar] = (
        time * num_rows
    )  # Create a variable containing amount of time to shift
    # Now create shifted date
    log("Creating shifted date.")
    df[newdate] = [
        date + time_shift(int(shift)) for date, shift in zip(df[datevar], df[shiftvar])
    ]
    log("expand_time completed.")

    # Cleanup and exit
    return df  # .drop('Shift', axis=1)


def expand_months(df: pd.DataFrame, datevar: str = "Date", newdatevar: str = "Daily Date", trade_days: bool = True):
    """
    Takes a monthly dataframe and returns a daily (trade day or calendar day) dataframe.
    For each row in the input data, duplicates that row over each trading/calendar day in the month of
    the date in that row. Creates a new date column containing the daily date.

    :Notes:

    If the input dataset has multiple observations per month, all of these will be expanded. Therefore
    you will have one row for each trade day for each original observation.

    :param df: DataFrame containing a date variable
    :param datevar: name of column containing dates in the input df
    :param newdatevar: name of new column to be created containing daily dates
    :param trade_days: True to use trading days and False to use calendar days
    :return:
    """
    if trade_days:
        td = tradedays()
    else:
        td = "D"

    expand = functools.partial(_expand, datevar=datevar, td=td, newdatevar=newdatevar)

    expand_all = np.vectorize(expand, otypes=[np.ndarray])

    days = pd.DataFrame(
        np.concatenate(expand_all(df[datevar].unique()), axis=0),
        columns=[datevar, newdatevar],
        dtype="datetime64",
    )

    return df.merge(days, on=datevar, how="left")


def tradedays():
    """
    Used for constructing a range of dates with pandas date_range function.

    :Example:

        >>> import pandas as pd
        >>> import pd_utils
        >>> pd.date_range(
        >>>     start='1/1/2000',
        >>>     end='1/31/2000',
        >>>     freq=pd_utils.tradedays()
        >>> )
        pd.DatetimeIndex(['2000-01-03', '2000-01-04', '2000-01-05', '2000-01-06',
                   '2000-01-07', '2000-01-10', '2000-01-11', '2000-01-12',
                   '2000-01-13', '2000-01-14', '2000-01-18', '2000-01-19',
                   '2000-01-20', '2000-01-21', '2000-01-24', '2000-01-25',
                   '2000-01-26', '2000-01-27', '2000-01-28', '2000-01-31'],
                  dtype='datetime64[ns]', freq='C')

    """
    trading_calendar = USTradingCalendar()
    return CustomBusinessDay(holidays=trading_calendar.holidays())


class USTradingCalendar(AbstractHolidayCalendar):
    """
    The US trading day calendar behind the function :py:func:`tradedays`.
    """
    rules = [
        Holiday("NewYearsDay", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("USIndependenceDay", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


def _expand(monthly_date, datevar, td, newdatevar):

    t = time.gmtime(monthly_date/1000000000) #date coming in as integer, need to parse
    t = datetime.date(t.tm_year, t.tm_mon, t.tm_mday) #better output than gmtime

    beginning = datetime.date(t.year, t.month, 1) #beginning of month of date
    end = beginning + relativedelta(months=1, days=-1) #last day of month
    days = pd.date_range(start=beginning, end=end, freq=td) #trade days within month
    days.name = newdatevar
    result =  np.array([(t, i) for i in days])
    return result


def year_month_from_single_date(date):
    d = OrderedDict()
    d.update({'Year': date.year})
    d.update({'Month': date.month})
    return pd.Series(d)