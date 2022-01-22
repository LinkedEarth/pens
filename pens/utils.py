import numpy as np
from datetime import datetime, timedelta
import cftime
from termcolor import cprint

def p_header(text):
    return cprint(text, 'cyan', attrs=['bold'])

def p_hint(text):
    return cprint(text, 'grey', attrs=['bold'])

def p_success(text):
    return cprint(text, 'green', attrs=['bold'])

def p_fail(text):
    return cprint(text, 'red', attrs=['bold'])

def p_warning(text):
    return cprint(text, 'yellow', attrs=['bold'])

def ymd2year_float(year, month, day):
    ''' Convert a set of (year, month, day) to an array of floats in unit of year
    '''
    year_float = []
    for y, m, d in zip(year, month, day):
        date = datetime(year=y, month=m, day=d)
        fst_day = datetime(year=y, month=1, day=1)
        lst_day = datetime(year=y+1, month=1, day=1)
        year_part = date - fst_day
        year_length = lst_day - fst_day
        year_float.append(y + year_part/year_length)

    year_float = np.asarray(year_float)
    return year_float

def datetime2year_float(date):
    ''' Convert a list of dates to floats in year
    '''
    if isinstance(date[0], np.datetime64):
        date = pd.to_datetime(date)

    year = [d.year for d in date]
    month = [d.month for d in date]
    day = [d.day for d in date]

    year_float = ymd2year_float(year, month, day)

    return year_float

def year_float2datetime(year_float, resolution='day'):
    if np.min(year_float) < 0:
        raise ValueError('Cannot handel negative years. Please truncate first.')

    ''' Convert an array of floats in unit of year to a datetime time; accuracy: one day
    '''
    year = np.array([int(y) for y in year_float], dtype=int)
    month = np.zeros(np.size(year), dtype=int)
    day = np.zeros(np.size(year), dtype=int)

    for i, y in enumerate(year):
        fst_day = datetime(year=y, month=1, day=1)
        lst_day = datetime(year=y+1, month=1, day=1)
        year_length = lst_day - fst_day

        year_part = (year_float[i] - y)*year_length + timedelta(minutes=1)  # to fix the numerical error
        date = year_part + fst_day
        month[i] = date.month
        day[i] = date.day

    if resolution == 'day':
        time = [cftime.datetime(y, m, d, 0, 0, 0, 0, 0, 0, has_year_zero=True) for y, m, d in zip(year, month, day)]
    elif resolution == 'month':
        time = [cftime.datetime(y, m, 1, 0, 0, 0, 0, 0, 0, has_year_zero=True) for y, m in zip(year, month)]

    return time

def year_float2dates(year_float):
    if np.min(year_float) < 0:
        raise ValueError('Cannot handel negative years. Please truncate first.')

    ''' Convert an array of floats in unit of year to a datetime time; accuracy: one day
    '''
    year = np.array([int(y) for y in year_float], dtype=int)
    dates = []

    for i, y in enumerate(year):
        fst_day = datetime(year=y, month=1, day=1)
        lst_day = datetime(year=y+1, month=1, day=1)
        year_length = lst_day - fst_day

        year_part = (year_float[i] - y)*year_length + timedelta(minutes=1)  # to fix the numerical error
        date = year_part + fst_day
        dates.append(date)


    return dates