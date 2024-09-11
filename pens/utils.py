import numpy as np
from scipy.special import rel_entr
import statsmodels.api as sm
from statsmodels.tsa.arima_process import arma_acf
#import pyleoclim as pyleo
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

# def model_acf(model, param, max_lag):        
#      '''
#      Generates the autocorrelation function (ACF) of a timeseries model given its parameters. 
     
#      Parameters
#      ----------
     
#      model : str
#          Name of the stochastic model describing the temporal behavior. Accepted choices are:

#          - `ar`: autoregressive model, see  https://www.statsmodels.org/dev/tsa.html#univariate-autoregressive-processes-ar
#          - `fGn`: fractional Gaussian noise, see https://stochastic.readthedocs.io/en/stable/noise.html#stochastic.processes.noise.FractionalGaussianNoise 
#          - `power-law`: aka Colored Noise, see https://stochastic.readthedocs.io/en/stable/noise.html#stochastic.processes.noise.ColoredNoise
         
#      param : variable type 
#          parameter of the model. 

#          - `ar`: param is the result from fitting with Statsmodels Autoreg.fit()
#          - `fGn`: param is the Hurst exponent, H (float)
#          - `power-law`: param is the spectral exponent beta (float)
         
#          Under allowable values, `fGn` and `power-law` should return equivalent results as long as H = (beta+1)/2 and H in [0, 1)
      
#      max_lag : int
#          maximum lag    

#      Returns
#      -------
#      ACF : numpy array, length max_lag (zero is included)                                                                                                            
      
#      '''
#      k = np.arange(max_lag)  # vector of lags
#      if model == 'power-law':  # https://www.dsprelated.com/showarticle/40.php
#          if param>1:
#              raise ValueError('β>1 will result in nonstationary autocovariance. Use a different model/parameter')
#          else:
#              acf = k**(param-1)  # what normalization ?
#          acf = k**(param-1)
#      elif model == 'fGn':
#          H = param
#          acf = 0.5*(np.abs(k+1)**(2*H) + np.abs(k-1)**(2*H) - 2 * np.abs(k)**(2*H))  
#      elif model == 'ar':
#          arparams = np.r_[1, -param[1:]]
#          maparams = np.r_[1, np.zeros_like(param)]
#          acf = arma_acf(ar = arparams, ma = maparams,lags=max_lag)
        
#      return acf   
             
     
     


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
