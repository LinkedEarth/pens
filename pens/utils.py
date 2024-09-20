import numpy as np
from scipy.special import rel_entr
import statsmodels.api as sm
from statsmodels.tsa.arima_process import arma_acf
from tqdm import tqdm

def kl_div(p, q):
    # to avoid negative values
    p = np.exp(p)
    q = np.exp(q)

    # convert to probability distributions
    p /= np.sum(p)
    q /= np.sum(q)

    # calculate the KL divergence
    vec = rel_entr(p, q)
    res = np.sum(vec)
    return res

def hdi1d(ary, hdi_prob, skipna=True):
    '''Compute highest density interval over a 1d array.
    h/t: Arviz code: https://arviz-devs.github.io/arviz/_modules/arviz/stats/stats.html#hdi
    
    ary : NumPy array
        values over which to compute HDI
        
    hdi_prob : float
        probability 
        
    skipna : bool
        flag to decide whether to drop NaNs (defaults to True)
    
    '''
    if len(ary.shape) > 1:
        raise ValueError("Array must be 1-dimensional")
    
    if skipna:
        nans = np.isnan(ary)
        if not nans.all():
            ary = ary[~nans]
    n = len(ary)

    ary = np.sort(ary)
    interval_idx_inc = int(np.floor(hdi_prob * n))
    n_intervals = n - interval_idx_inc
    interval_width = np.subtract(ary[interval_idx_inc:], ary[:n_intervals], dtype=np.float_)

    if len(interval_width) == 0:
        raise ValueError("Too few elements for interval calculation. ")

    min_idx = np.argmin(interval_width)
    hdi_min = ary[min_idx]
    hdi_max = ary[min_idx + interval_idx_inc]

    hdi_interval = np.array([hdi_min, hdi_max])

    return hdi_interval

def means_and_trends_ensemble(var,segment_length,step,years):
    ''' Calculates the means and trends on an ensemble array
        Uses statsmodels' OLS method

     Inputs:
        var:                2d numpy array [time, ens member]
        segment_length:     # elements in block (integer)
        step:               step size (integer)
        years:              1d numpy array

     Outputs:
        means:      Means of every segment.
        trends:     trends over every segment.
        idxs:       The first and last index of every segment, for record-keeping.
        tm:         median time point of each block

    Author: Julien Emile-Geay, based on code by Michael P. Erb.
    Date: March 8, 2018
    '''
    try:
        import pyleoclim as pyleo
    except:
        raise ImportError('Need to install Pyleoclim: `pip install pyleoclim`')

    if var.ndim < 2:
       print("Beef up your ensemble, yo. Ain't got nothing in it")

    n_years, n_ens = var.shape
    if segment_length +2*step >= n_years:
        raise ValueError(f'Too few samples ({n_years}) for a segment length of {segment_length}')

    n_segments = int(((n_years-segment_length)/step)+1)
    skip_idx = np.remainder(n_years-segment_length,step)  # If the segments don't cover the entire time-span, skip years at the beginning.

    # Initialize vars
    means      = np.empty((n_segments,n_ens))
    trends     = np.empty((n_segments,n_ens))
    tm         = np.empty((n_segments))
    idxs       = np.empty((n_segments,2), dtype=int)

    fc = 1/segment_length #smoothing length

    # Compute the means and trends for every location
    for m in tqdm(range(n_ens), desc='Processing member'):
        y = var[:,m]
        yl = pyleo.utils.filter.butterworth(y,fc)

        for k in range(n_segments):
            start_idx = skip_idx+(k*step)
            end_idx   = start_idx+segment_length-1
            # compute block averages
            means[k,m] = np.mean(y[start_idx:end_idx])

            # compute trends with robust fit
            x = years[start_idx:end_idx]
            x = sm.add_constant(x)
            _, trends[k,m] = sm.OLS(yl[start_idx:end_idx],x).fit().params

            # indices
            if m == 0:
                idxs[k,:] = start_idx,end_idx-1
                tm[k]     = np.median(years[start_idx:end_idx])

    return means, trends, tm, idxs

def standardize(x, scale=1, axis=0, ddof=0, eps=1e-3):
    """Centers and normalizes a time series. Constant or nearly constant time series not rescaled.

    Parameters
    ----------
    x : array
        vector of (real) numbers as a time series, NaNs allowed
    scale : real
        A scale factor used to scale a record to a match a given variance
    axis : int or None
        axis along which to operate, if None, compute over the whole array
    ddof : int
        degress of freedom correction in the calculation of the standard deviation
    eps : real
        a threshold to determine if the standard deviation is too close to zero

    Returns
    -------
    z : array
       The standardized time series (z-score), Z = (X - E[X])/std(X)*scale, NaNs allowed
    mu : real
        The mean of the original time series, E[X]
    sig : real
         The standard deviation of the original time series, std[X]

    References
    ----------

    Tapio Schneider's MATLAB code: https://github.com/tapios/RegEM/blob/master/standardize.m

    The zscore function in SciPy: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.zscore.html

    See also
    --------

    pyleoclim.utils.tsutils.preprocess : pre-processes a times series using standardization and detrending.

    """
    x = np.asanyarray(x)
    assert x.ndim <= 2, 'The time series x should be a vector or 2-D array!'

    mu = np.nanmean(x, axis=axis)  # the mean of the original time series
    sig = np.nanstd(x, axis=axis, ddof=ddof)  # the standard deviation of the original time series

    mu2 = np.asarray(np.copy(mu))  # the mean used in the calculation of zscore
    sig2 = np.asarray(np.copy(sig) / scale)  # the standard deviation used in the calculation of zscore

    if np.any(np.abs(sig) < eps):  # check if x contains (nearly) constant time series
        warnings.warn('Constant or nearly constant time series not rescaled.',stacklevel=2)
        where_const = np.abs(sig) < eps  # find out where we have (nearly) constant time series

        # if a vector is (nearly) constant, keep it the same as original, i.e., substract by 0 and divide by 1.
        mu2[where_const] = 0
        sig2[where_const] = 1

    if axis and mu.ndim < x.ndim:
        z = (x - np.expand_dims(mu2, axis=axis)) / np.expand_dims(sig2, axis=axis)
    else:
        z = (x - mu2) / sig2

    return z, mu, sig