#from multiprocessing.sharedctypes import Value
#from matplotlib import gridspec
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import copy
import sklearn.metrics
from tqdm import tqdm
from . import utils
#import statsmodels as sm
import scipy.linalg as linalg
from scipy.stats import gaussian_kde 
from scipy.stats import mode
#from scipy.optimize import curve_fit
from scipy.stats import percentileofscore 
#from scipy.stats import multivariate_normal
from pyleoclim.utils.tsutils import standardize
from statsmodels.tsa.arima_process import arma_generate_sample
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#import covar
import properscoring as ps
from more_itertools import distinct_combinations


class EnsembleTS:
    ''' Ensemble Timeseries

    Note that annual reconstruction is assumed so the time axis is in years.
    The ensembles variable should be in shape of (nt, nEns), where nt is the number of years,
    and nEns is the number of ensemble members.

    '''
    def __init__(self, time=None, value=None, label=None,  time_name=None, 
                 time_unit=None, value_name=None, value_unit=None):
        if np.ndim(value) == 1:
            value = value[:, np.newaxis]

        self.time = time
        self.value = value
        self.label = label
        self.time_name = time_name
        self.time_unit = time_unit
        self.value_name = value_name
        self.value_unit = value_unit

        if self.value is not None:
            self.refresh()
            
    def make_labels(self):
        '''
        Initialization of plot labels based on object metadata

        Returns
        -------
        time_header : str
            Label for the time axis
        value_header : str
            Label for the value axis

        '''
        if self.time_name is not None:
            time_name_str = self.time_name
        else:
            time_name_str = 'time'

        if self.value_name is not None:
            value_name_str = self.value_name
        else:
            value_name_str = 'value'

        if self.value_unit is not None:
            value_header = f'{value_name_str} [{self.value_unit}]'
        else:
            value_header = f'{value_name_str}'

        if self.time_unit is not None:
            time_header = f'{time_name_str} [{self.time_unit}]'
        else:
            time_header = f'{time_name_str}'

        return time_header, value_header 

    def refresh(self):
        self.nt = np.shape(self.value)[0]
        self.nEns = np.shape(self.value)[1]
        self.median = np.nanmedian(self.value, axis=1)
        self.mean = np.nanmean(self.value, axis=1)
        self.std = np.nanstd(self.value, axis=1)
        
    def subsample(self, nsamples, seed=None):
        '''
        Thin out original ensemble by drawing nsamples at random

        Parameters
        ----------
        nsamples : int
            number of samples to draw at random from the original ensemble.
            If nsamples >= self.nEns, the object is returned unchanged.
            
        seed : int
            seed for the random generator (provided for reproducibility)

        Returns
        -------
        res : EnsembleTS 
            Downsized object.

        '''
        if seed is not None:
            np.random.seed(seed)
        
        if nsamples < self.nEns:     
            res = self.copy() # copy object to get metadata
            res.value = self.value[:,np.random.randint(low=0, high=self.nEns, size=nsamples)] # subsample
            res.refresh()
        else:
            res = self
        return res 

    def get_mean(self):
        res = self.copy() # copy object to get metadata
        res.value = self.mean[:, np.newaxis]
        res.refresh()
        return res
    
    def get_mode(self):
        res = self.copy() # copy object to get metadata
        md, counts = mode(self.value, axis=1, nan_policy='raise', keepdims=True)
        res.value = md
        res.refresh()
        return res

    def get_median(self):
        res = self.copy() # copy object to get metadata
        res.value = self.median[:, np.newaxis]
        res.refresh()
        return res

    def get_std(self):
        res = self.copy() # copy object to get metadata
        res.value = self.std[:, np.newaxis]
        res.refresh()
        return res
    
    def get_means_and_trends(self, segment_length=10, step=10, xm=np.linspace(-0.5,1.5,200), bw = 'silverman'):
        '''
        Extract trend distributions from EnsembleTS object via Gaussian Kernel Density Estimation

        Parameters
        ----------
        segment_length : int, optional
            DESCRIPTION. The default is 10.
        step : int, optional
            DESCRIPTION. The default is 10.
        xm : NumPy array, optional
            axis over which KDE is calculated The default is np.linspace(-0.5,1.5,200).
        bw : str, scalar or callable, optional
            The method used to calculate the estimator bandwidth. 
            This can be "scott", "silverman", a scalar constant or a callable.
            If a scalar, this will be used directly as kde.factor.
            If a callable, it should take a gaussian_kde instance as only parameter and return a scalar.
            If None (default), "scott" is used. 

        Returns
        -------
        new : TYPE
            DESCRIPTION.

        '''
        new = self.copy()
        means, trends, tm, idxs = utils.means_and_trends_ensemble(self.value, segment_length, step, self.time)
        dmeans  = means[-1:] - means[:-1] # difference of means
        dtrends = segment_length*(trends[-1:] - trends[:-1]) # difference of trends

        dm_kde = gaussian_kde(dmeans.flatten(),bw_method=bw)
        dm_prob = dm_kde.integrate_box_1d(0, xm.max()) # estimate probability of positive change
        dt_kde = gaussian_kde(dtrends.flatten(),bw_method=bw)
        dt_prob = dt_kde.integrate_box_1d(0, xm.max()) # estimate probability of positive change
        res_dict = {
            'means': means,
            'trends': trends,
            'tm': tm,
            'idxs': idxs,
            'dm_prob': dm_prob,
            'dt_prob': dt_prob,
            'dm_kde': dm_kde,
            'dt_kde': dt_kde,
            'xm': xm,
        }
        new.trend_dict = res_dict
        return new

    def __getitem__(self, key):
        ''' Get a slice of the ensemble.
        '''
        new = self.copy()
        new.value = new.value[key]
        if type(key) is tuple:
            new.time = new.time[key[0]]
        else:
            new.time = new.time[key]

        new.refresh()
        return new

    def __add__(self, series):
        ''' Add a series to the ensemble.

        Parameters
        ----------
        series : int, float, array, EnsembleTS
            A series to be added to the value field of each ensemble member.
            Can be a constant int/float value, an array, or another EnsembleTS object with only one member.
            If it's an EnsembleTS that has multiple members, the median will be used as the series.

        '''
        new = self.copy()
        if isinstance(series, EnsembleTS):
            series = series.median

        if np.ndim(series) > 0:
            series = np.array(series)[:, np.newaxis]

        new.value += series
        new.refresh()
        return new

    def __sub__(self, series):
        ''' Substract a series from the ensemble.

        Parameters
        ----------
        series : int, float, array, EnsembleTS
            A series to be substracted from the value field of each ensemble member.
            Can be a constant int/float value, an array, or another EnsembleTS object with only one member.
            If it's an EnsembleTS that has multiple members, the median will be used as the series.
        '''
        new = self.copy()
        if isinstance(series, EnsembleTS):
            series = series.median

        if np.ndim(series) > 0:
            series = np.array(series)[:, np.newaxis]

        new.value -= series
        new.refresh()
        return new

    def __mul__(self, series):
        ''' Element-wise multiplication. The multiplier should have the same length as self.nt.

        Parameters
        ----------
        series : int, float, array, EnsembleTS
            A series to be element-wise multiplied by for the value field of each ensemble member.
            Can be a constant int/float value, an array, or another EnsembleTS object with only one member.
            If it's an EnsembleTS that has multiple members, the median will be used as the series.
        '''
        new = self.copy()
        if isinstance(series, EnsembleTS):
            series = series.median

        if np.ndim(series) > 0:
            for i in range(self.nt):
                new.value[i] *= series[i]
        else:
            new.value *= series

        new.refresh()
        return new

    def __truediv__(self, series):
        ''' Element-wise division. The divider should have the same length as self.nt.

        Parameters
        ----------
        series : int, float, array, EnsembleTS
            A series to be element-wise divided by for the value field of each ensemble member.
            Can be a constant int/float value, an array, or another EnsembleTS object with only one member.
            If it's an EnsembleTS that has multiple members, the median will be used as the series.
        '''
        new = self.copy()
        if isinstance(series, EnsembleTS):
            series = series.median

        if np.ndim(series) > 0:
            for i in range(self.nt):
                new.value[i] /= series[i]
        else:
            new.value /= series

        new.refresh()
        return new

    def copy(self):
        return copy.deepcopy(self)

    
    def slice(self, timespan):
        ''' Slicing the timeseries with a timespan (tuple or list)

        Parameters
        ----------
        timespan : tuple or list
            The list of time points for slicing, whose length must be even.
            When there are n time points, the output Series includes n/2 segments.
            For example, if timespan = [a, b], then the sliced output includes one segment [a, b];
            if timespan = [a, b, c, d], then the sliced output includes segment [a, b] and segment [c, d].

        Returns
        -------
        new : EnsembleTS
            The sliced EnsembleSeries object.
        '''
        new = self.copy()
        
        n_elements = len(timespan)
        if n_elements % 2 == 1:
            raise ValueError('The number of elements in timespan must be even!')

        n_segments = int(n_elements / 2)
        mask = [False for i in range(np.size(self.time))]
        for i in range(n_segments):
            mask |= (self.time >= timespan[i*2]) & (self.time <= timespan[i*2+1])

        new.time=self.time[mask]
        new.value=self.value[mask]
        new.refresh()
        
        return new
    
    def load_nc(self, path, time_name='time', var=None):
        ''' Load data from a .nc file with xarray

        Parameters
        ----------
        path : str
            The path of the .nc file.

        var : str
            The name of variable to load.
            Note that we assume the first axis of the loaded variable is time.

        time_name : str
            The name of the time axis.

        '''
        ds = xr.open_dataset(path)
        if time_name == 'year':
            time = ds[time_name].values
        else:
            time = np.array([t.year for t in ds[time_name].values])

        arr = ds[var].values
        if ds[var].dims.index(time_name) != 0:
            arr = np.moveaxis(arr, ds[var].dims.index(time_name), 0)

        nt = len(time)
        value = np.reshape(arr, (nt, -1))

        new = EnsembleTS(time=time, value=value)
        return new

    def from_df(self, df, time_column=None, value_columns=None):
        ''' Load data from a pandas.DataFrame

        Parameters
        ----------
        df : pandas.DataFrame
            The pandas.DataFrame object.

        time_column : str
            The label of the column for the time axis.

        value_columns : list of str
            The list of the labels for the value axis of the ensemble members.

        '''
        if time_column is None:
            raise ValueError('`time_column` must be specified!')

        if value_columns is None:
            value_columns = list(set(df.columns) - {time_column})
            
        arr = df[value_columns].values
        time = df[time_column].values
        nt = len(time)
        value = np.reshape(arr, (nt, -1))

        ens = EnsembleTS(time=time, value=value)
        return ens

    def to_df(self, time_column=None, value_column='ens'):
        ''' Convert an EnsembleTS to a pandas.DataFrame

        Parameters
        ----------
        time_column : str
            The label of the column for the time axis.

        value_column : str
            The base column label for the ensemble members.
            By default, the columns for the members will be labeled as "ens.0", "ens.1", "ens.2", etc.

        '''
        time_column = 'time' if time_column is None else time_column
        data_dict = {}
        data_dict[time_column] = self.time
        nt, nEns = np.shape(self.value)
        for i in range(nEns):
            data_dict[f'{value_column}.{i}'] = self.value[:, i]

        df = pd.DataFrame(data_dict)
        return df

    def to_pyleo(self, **kwargs):
        ''' Convert to a `pyleoclim.EnsembleSeries` or `pyleoclim.Series` object

        Parameters
        ----------
        kwargs : keyword arguments
            keyword arguments for a `pyleoclim.Series` object

        '''
        try:
            import pyleoclim as pyleo
        except:
            raise ImportError('Need to install Pyleoclim: `pip install pyleoclim`')

        series_list = []
        for i in range(self.nEns):
            ts = pyleo.Series(time=self.time, value=self.value[..., i], **kwargs)
            series_list.append(ts)

        if len(series_list) == 1:
            es = ts
        else:
            es = pyleo.EnsembleSeries(series_list)   
        
        # transfer metadata
        es.value_name = self.value_name 
        es.value_unit = self.value_unit
        es.time_name = self.time_name
        es.time_unit = self.time_unit
        
        return es

    def random_paths(self, model='fGn', param = None, p = 1, trend = None, seed=None):
        ''' 
        Generate `p` random walks through the ensemble according to a given
        parametric model with random parameter sampling
            
        Parameters
        ----------
        
        model : str
            Stochastic model for the temporal behavior. Accepted choices are:

            - `unif`: resample uniformly from the posterior distribution
            - `ar`: autoregressive model, see  https://www.statsmodels.org/dev/tsa.html#univariate-autoregressive-processes-ar
            - `fGn`: fractional Gaussian noise, see https://stochastic.readthedocs.io/en/stable/noise.html#stochastic.processes.noise.FractionalGaussianNoise 
            - `power-law`: aka Colored Noise, see https://stochastic.readthedocs.io/en/stable/noise.html#stochastic.processes.noise.ColoredNoise
            
        param : variable type [default is None]
            parameter of the model. 

            - `unif`: no parameter 
            - `ar`: param is the result from fitting Statsmodels Autoreg.fit() (with zero-lag term)
            - `fGn`: param is the Hurst exponent, H (float)
            - `power-law`: param is the spectral exponent beta (float)
            
            Under allowable values, `fGn` and `power-law` should return equivalent results as long as H = (beta+1)/2 is in [0, 1)
            
        p : int
            number of series to export
            
        trend : array, length self.nt
            general trend of the ensemble. 
            If None, it is calculated as the ensemble mean.
            If provided, it will be added to the ensemble. 
              
        seed : int
            seed for the random generator (provided for reproducibility)

        Returns
        -------
        new :  EnsembleTS object containing the p series 
            
        '''
        if seed is not None:
            np.random.seed(seed)
        if trend is None:
            trend = self.get_mean()
            
        N = self.nt                            
        scale = self.get_std()       
            
        paths = np.ndarray((N, p))
        
        if model == 'unif':
            idx = np.random.randint(low=0, high=self.nEns, size=(self.nt, p))      
            for ie in range(p):
                for it in range(self.nt):
                    paths[it, ie] = self.value[it, idx[it, ie]]
                    
        elif model == 'power-law':
            from stochastic.processes.noise import ColoredNoise
            for j in tqdm(range(p)):
                CN = ColoredNoise(beta=param,t=N)
                z, _, _ = standardize(CN.sample(N-1))
                paths[:,j] = z
                 
        elif model == 'fGn':
            from stochastic.processes.noise import FractionalGaussianNoise
            for j in tqdm(range(p)):
                fgn = FractionalGaussianNoise(hurst=param, t=N)
                z, _, _ = standardize(fgn.sample(N, algorithm='daviesharte')) 
                paths[:,j] = z
            
        elif model == 'ar':
            # TODO: enable random sampling of parameters
            coeffs = param[1:] # ignore the zero-lag term
            arparams = np.r_[1, -coeffs]
            maparams = np.r_[1, np.zeros_like(coeffs)]
           
            for j in tqdm(range(p)):
                y = arma_generate_sample(arparams, maparams, N)
                z, _, _ = standardize(y)
                paths[:,j] = z
          
        new = self.copy()                 
        new.value = paths
        new.nEns = p 
        if self.label is not None:
            new.label = f'{self.label} ({model} resampling)'
        else:
            new.label = f'{model} resampling'
        
        if model != 'unif':
            scale = self.get_std()
            trend = self.get_mean()
            new = new*scale + trend # add the trend back in
        
        return new

    def sample_nearest(self, target, metric='MSE'):
        ''' Get the nearest sample path against the target series
        
        Note that `metric` is used only for the final distance calculation.
        '''
        dist_func = {
            'MSE': sklearn.metrics.mean_squared_error,
            'KLD': utils.kl_div,
        }

        path = np.ndarray((self.nt, 1))
        target_idx = []
        for it in range(self.nt):
            im = np.argmin(np.abs(self.value[it]-target[it]))
            target_idx.append(im)
            path[it] = self.value[it, im]

        new = EnsembleTS(time=self.time, value=path)
        new.distance = dist_func[metric](new.value[:, 0], target)
        new.target = target
        new.target_idx = target_idx

        return new
        

    # def compare_nearest(self, ens, metric='MSE'):
    #     ''' Compare with the nearest path from another EnsembleTS

    #     Note that we assume the size of the time axis is consistent.
    #     If not, please call EnsembleTS.slice() ahead.
    #     '''
    #     dist = np.zeros(ens.nEns)
    #     for i in tqdm(range(ens.nEns)):
    #         target = ens.value[:, i]
    #         dist[i] = self.sample_nearest(target, metric=metric).distance

    #     return dist

    # def compare(self, ens, metric='MSE'):
    #     ''' Compare with another EnsembleTS

    #     Note that we assume the size of the time axis is consistent.
    #     If not, please call EnsembleTS.slice() ahead.
    #     '''
    #     dist_func = {
    #         'MSE': sklearn.metrics.mean_squared_error,
    #         'KLD': utils.kl_div,
    #     }
    #     max_nens = np.min([self.nEns, ens.nEns])
    #     dist = np.zeros(ens.nt)
    #     for i in tqdm(range(ens.nt)):
    #         dist[i] = dist_func[metric](self.value[i, :max_nens], ens.value[i, :max_nens])

    #     return dist
    

    def crps_trace(self, y):
        '''
        Computes the continuous ranked probability score (CRPS) of trace y against the ensemble.
        This view flips the traditional definition of the CRPS (Matheson and Winkler 1976; Hersbach 2000)
        as verifying a forecast ensemble against a deterministic observation y. 
        
        
        Parameters
        ----------
        y : array-like, length n
            trace whose consistency with the ensemble is to be assessed
            Must have n == self.nt. 

        Returns
        -------
        crps : array-like, length n

        '''
        if len(y) != self.nt:
            raise ValueError('the target series and the ensemble must have the same length')
           
        crps = np.zeros_like(y)
        for i in range(self.nt):
            crps[i] = ps.crps_ensemble(y[i], self.value[i,:])
            
        return crps
    
    def trace_rank(self, y):
        '''
        Computes ensemble rank (expressed as percentile) for trace y
        
        Parameters
        ----------
        y : array-like, length n
            trace whose rank within the ensemble is to be assessed
            Must have n == self.nt. 
        
        Returns
        -------
        percent : array-like, length n
        
        '''
        if len(y) != self.nt:
            raise ValueError('the target series and the ensemble must have the same length')
        
        percent = np.zeros_like(y)
        for k in range(self.nt):
            percent[k] = percentileofscore(self.value[k,:],y[k])

        return percent
       

    def hdi_score(self, y, prob=0.9):
        '''
        Computes HDI score for target series y
        
        Parameters
        ----------
        y : array-like, length n
            trace whose intensity of probability ("likelihood") is to be assessed
            Must have n == self.nt. 
        
        prob : float
            probability for which the highest density interval will be computed. The default is 0.9.
            
        Returns
        -------
        score: the score (scalar)
        HDI : the n x 2 array
        
        '''
        if len(y) != self.nt:
            raise ValueError('the target series and the ensemble must have the same length')
        
        HDI = np.zeros((self.nt,2))
        sc = np.zeros(self.nt)
        
        for it in range(self.nt):
            HDI[it,:] = utils.hdi1d(self.value[it,:], hdi_prob=prob)
            if (y[it] >= HDI[it,0]) and (y[it] <= HDI[it,1]):
                sc[it]=1
        # normalize
        score = sc.mean()
        
        return score, HDI
    
    def distance(self, y=None, order=1, nsamples=None):
        '''
        Compute the distance between a target y and the ensemble object.
        
        Parameters
        ----------
        y : array-like, length n
            trace/plume whose probability is to be assessed
            If None, the distance is computed between every possible pair of trajectories within the ensemble
            If specified, Must have n == self.nt
    
        order : int, or inf
            Order of the norm. inf means numpy’s inf object. The default is 1.
            
        nsamples : int
            number of samples to use (to speed up computation for very large ensembles)
        
        See Also
        --------
        np.linalg.norm: https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html

        Returns
        -------
        dist : numpy array, dimension (self.nEns,)
            
        '''
        if nsamples is not None:
            left = self.subsample(nsamples=nsamples)
        else:
            left = self.copy()
            
            
        if y is None:  # compute intra ensemble distance, pairwise
           candidate_pairs = distinct_combinations(np.arange(left.nEns), 2)
           dist = []
           for p in tqdm(list(candidate_pairs),desc='Computing intra-ensemble distance among possible pairs'):
               series_diff = left.value[:,p[0]]-left.value[:,p[1]]
               dist.append(np.linalg.norm(series_diff, ord = order)/left.nt)
           d = np.array(dist)    
        else:
            if isinstance(y, EnsembleTS):
                if nsamples is not None:
                    right = y.subsample(nsamples=nsamples)
                    y = right.value  # extract NumPy array
                else:
                    y = y.value
                
            if y.shape[0] != self.nt:
                raise ValueError('the target series and the ensemble must have the same length')
            # handle plume dimension 
            if len(y.shape) > 1:
                ncol = y.shape[1]
                d = np.zeros((left.nEns,ncol))
                for i in tqdm(range(left.nEns),desc='Computing inter-ensemble distance'):
                    for j in range(ncol):
                        d[i,j] = np.linalg.norm(left.value[:,i]-y[:,j], ord = order)/left.nt
                d = np.reshape(d, (left.nEns*ncol))
            else:
                d = np.zeros((left.nEns))
                for i in range(left.nEns):
                    d[i] = np.linalg.norm(left.value[:,i]-y, ord = order)/self.nt
            
        return d
    
    def proximity_prob(self, y, eps, order=1, dist=None, nsamples=None):
        '''
        Compute the probability P that the trace y is within a distance eps of the ensemble object.
        
        Parameters
        ----------
        y : array-like, length self.nt
            trace/plume whose proximity is to be assessed
        eps : array of float64
            numerical tolerance for the distance. 
        order : int, or inf
            Order of the norm. inf means numpy’s inf object. The default is 1.
        dist : array-like, length self.nEns
            if provided, uses this as vector of distances. Otherwise it is computed internally
            
        See Also
        --------
        np.linalg.norm: https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html

        Returns
        -------
        P : float in [0,1]
            Probability that the trace y is within a distance eps of the ensemble object

        '''
        if dist is None:
            dist = self.distance(y, order=order, nsamples=nsamples)
            
        if isinstance(eps, np.ndarray): 
            prob = np.zeros_like(eps)
            for j, tol in enumerate(eps):
                prob[j] = np.where(dist<=tol)[0].size/dist.size
        elif isinstance(eps, float):
            prob = np.where(dist<=eps)[0].size/dist.size
        else:
            raise ValueError('eps is of unsupported type '+ str(type(eps)))
            
        if any(prob > 1):
            raise ValueError("Did you know? Probabilities are bounded by unity.")
            
        return prob
         
    def plume_distance(self, y=None, max_dist=1, num=100, q = 0.5, order=1, dist=None, nsamples=None):
        '''
        Compute the (quantile-based) characteristic distance between 
        a plume (ensemble) and another object (whether a single trace or another plume).
        Searches for quantile q of the "proximity probability" distribution
        
        Parameters
        ----------
        y : array-like, length self.nt
           trace/plume whose probability is to be assessed
        max_dist : maximum distance to consider in the calculation of proximity probabilities
            default is 1, which may or may not make sense for your application!  
        num : int
            number of discrete points for the estimation of the distance distribution
        order : int, or inf
            Order of the norm. inf means numpy’s inf object. The default is 1.
        q : float
           Quantile from which the characteristic distance is derived. Default = 0.5 (median)     
        dist : array-like, length self.nEns
            if provided, uses this as vector of distances. Otherwise it is computed internally
            
        See Also
        --------
        np.linalg.norm: https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html

        Returns
        -------
        charac_eps: float 
            Representative distance (in same units as self or y)

        '''
        eps = np.linspace(0,max_dist,num=num) # vector of distances
        def find_roots(x,y):
            s = np.abs(np.diff(np.sign(y))).astype(bool)
            return x[:-1][s] + np.diff(x)[s]/(np.abs(y[1:][s]/y[:-1][s])+1)
        
        if y is None:  # compute intra ensemble distance
            d = self.distance(order=order, nsamples=nsamples)
            prob = self.proximity_prob(self.value, eps=eps, dist=d, nsamples=nsamples)
            charac_eps = find_roots(eps, prob - q)[0]
        else:  # compute plume distance to y (EnsembleTS or array)
            if isinstance(y, EnsembleTS):
                if y.nEns == self.nEns:
                    if np.allclose(y.value, self.value):
                        print('objects are numerically identical')
                        charac_eps = 0
                    else:
                        prob = self.proximity_prob(y.value, eps=eps, order=order, dist=dist, nsamples=nsamples)
                        charac_eps = find_roots(eps, prob - q)[0]
            else :
                # assess proximity probability between the ensemble and the object (trace or ensemble)
                prob = self.proximity_prob(y=y, eps=eps, order=order, dist=dist, nsamples=nsamples)
                charac_eps = find_roots(eps, prob - q)[0]

        return charac_eps
           
    def SmBP(self, y1, y2, acf, d=None):
        '''
        Computes the ratio of probability intensities based on the Small Ball Probability (SmBP)
        concept of [1, 2]. Because of normalization issues, this number is only meaningful in a 
        relative sense, i.e. when comparing how two traces y1 and y2 fit within an ensemble. 
        For Gaussian processes, the SmBP has the true interpretation of an intensity 
        of a probability, exactly like a true probability density function, 
        i.e. like an ordinary likelihood L.

        For such processes, L(ensemble mean) = 1, and any other trace has a 
        lower likelihood. Note that y1, y2 are internally standardized prior to analysis.
        
        Parameters
        ----------
        y1 : array-like, length n            
            Must have n == self.nt. 
            
        y2 : array-like, length n
            Must have n == self.nt.  
            
        acf : array-like, length n
            autocorrelation function associated with the model
        
        d : int, optional
            Truncation of the eigenvalue expansion of the ensemble's covariance.
            If no truncation is provided, all n eigenmodes are used. 
            If d is a vector of integers, a vector of likelihood ratios is returned. 
            
        References
        ----------
        [1]_ Bongiorno, E. G., and A. Goia (2017), Some insights about the small ball probability factorization for Hilbert random elements, Statistica Sinica, pp. 1949–1965, doi: 10.5705/ss.202016.0128.

        [2]_ Li, W. V., and Shao, Q. M. (2001), Gaussian processes: Inequalities, small ball probabilities and applications, vol. 19, pp. 533–597, Elsevier, doi:10.1016/S0169-7161(01)19019- X.
        
        Returns
        -------
        L(y2|X,d)/L(y1|X,d), the likelihood ratio of observing y2 vs y1, given X and d.  

        '''
    
        if len(y1) != self.nt or len(y2) != self.nt:
            raise ValueError("The series and ensemble must have the same time dimension")
            
        if d is None:
            d = self.nt
            print('No truncation provided. Using all ' +str(d)+ ' modes')
             
        ys1, mu1, std1 = standardize(y1)
        ys2, mu2, std2 = standardize(y2)
        
        # form covariance matrix
        Sigma = linalg.toeplitz(acf)
        
        # eigendecomposition 
        L, V = np.linalg.eigh(Sigma)
        eigvals = np.flipud(L) # change to decreasing order 
        eigvecs = np.flip(V,axis=1)
        
        # project y onto the eigenvectors
        yj1 = np.dot(ys1,eigvecs)
        yj2 = np.dot(ys2,eigvecs)
    
        # compute the SmBP
        Psi_1 = -0.5*np.square(yj1)/eigvals
        Psi_2 = -0.5*np.square(yj2)/eigvals
        loglik = np.sum(Psi_2[:d]) - np.sum(Psi_1[:d])
        lik = np.exp(loglik)
        
        # gather diagnostics
        #diag = {}
        #diag['var '] = np.cumsum(eigvals)/eigvals.sum()*100
                     
        return lik
        
    # def likelihood(self, target, acf, inv_method = 'pseudoinverse'):
    #     '''
    #     Computes the likelihood of observing a target trajectory conditional on 
    #     the ensemble's distribution (self).
    #     Removes deterministic trends and time-dependent scaling ; make sure the
    #     specified autocorrelation function (acf) corresponds to a model fit 
    #     under such assumptions.
        
    #     Parameters
    #     ----------
        
    #     target : Pyleoclim Series object
    #         Timeseries to be evaluated against ensemble EnsTS
            
    #     acf : array (same length as EnsTS and target)
    #         autocorrelation function associated with the model
        
    #     inv_method : str
    #         Method to use in inverting the covariance matrix
    #         Acceptable choices include:
    #         - Moore-Penrose inverse ('pseudoinverse') [default]
    #         - Chen, Wiesel, and Hero (2009) ('CWH09') see https://arxiv.org/pdf/1009.5331.pdf
    #     (the last one courtesy of the covar package: https://pythonhosted.org/covar/)

    #     Returns
    #     -------
    #     L : likelihood ratio between the target and the ensemble mean
    #     D : Mahalanobis distance to the ensemble mean
    #     '''
        
        
    #     y = target.value
    #     mu = self.get_mean().value[:,0]
    #     sig = self.get_std().value[:,0]
        
    #     #self = (self - mu)/sig
    #     ys = (y - mu)/sig

    #     if len(y) != self.nt:
    #         raise ValueError("The series and ensemble must have the same time dimension")
             
    #     # form covariance matrix
    #     Sigma = linalg.toeplitz(acf)
        
    #     # compute its inverse
    #     if inv_method == 'pseudoinverse':
    #         Sigma_i = linalg.pinv(Sigma)
    #     elif inv_method == 'CWH09':
    #         # first estimate the ACF e-folding scale to get DOFs
    #         monoExp = lambda x, A, tau: A*np.exp(-x/tau)
    #         lags = np.arange(self.nt)
    #         p0 = (1, 10) # start with values near those we expect
    #         params, cv = curve_fit(monoExp, lags, acf, p0)
    #         tau = params[1] # e-folding time
    #         dof =  0.5*self.nt/tau  # https://www.earthinversion.com/geophysics/estimation-degrees-of-freedom/
    #         Sigma_i, g  = covar.cov_shrink_rblw(Sigma,n=dof)  
    #         print('Optimal Covariance Shrinkage: {:3.2f}'.format(g))
     
    #     # Mahalanobis distance
    #     d2 = ys.T.dot(Sigma_i).dot(ys)
    #     D = np.sqrt(d2)
    #     # likelihood
    #     L = np.exp(-d2/2) 

    #     #dy = scipy.spatial.distance.mahalanobis(y,mu,Sigma_i)
        
    #     # compute the log PDFs (vectors are so large that the numbers are impossibly close to 0 otherwise)
    #     #log_y = multivariate_normal.logpdf(y, mean=0, cov=Sigma, allow_singular=True)
    #     #log_mu = multivariate_normal.logpdf(0, mean=0, cov=Sigma, allow_singular=True)
    #     #loglik = log_y-log_mu
        
        
    #     return L, D
            

   
    def line_density(self, figsize=[10, 4], cmap='Greys', color_scale='linear', bins=None, num_fine=None,
        xlabel= None, ylabel=None, title=None, ylim=None, xlim=None, 
        title_kwargs=None, ax=None, **pcolormesh_kwargs,):
        ''' Plot the timeseries 2-D histogram

        Parameters
        ----------
        cmap : str
            The colormap for the histogram.

        color_scale : str
            The scale of the colorbar; should be either 'linear' or 'log'.

        bins : list/tuple of 2 floats
            The number of bins for each axis: nx, ny = bins.

        Referneces
        ----------
        - https://matplotlib.org/3.6.0/gallery/statistics/time_series_histogram.html

        '''
        pcolormesh_kwargs = {} if pcolormesh_kwargs is None else pcolormesh_kwargs

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        if num_fine is None:
            num_fine = np.min([self.nt*8, 1000])

        num_series = self.nEns
        x = self.time
        Y = self.value.T
        x_fine = np.linspace(x.min(), x.max(), num_fine)
        y_fine = np.empty((num_series, num_fine), dtype=float)
        for i in range(num_series):
            y_fine[i, :] = np.interp(x_fine, x, Y[i, :])
        y_fine = y_fine.flatten()
        x_fine = np.tile(x_fine, [num_series, 1]).flatten()

        if bins is None:
            bins = [num_fine//2, num_series//10]

        h, xedges, yedges = np.histogram2d(x_fine, y_fine, bins=bins)
        h = h / h.max()  # normalize

        pcm_kwargs = {}
        # if 'vmax' in pcolormesh_kwargs:
        #     vmax = pcolormesh_kwargs['vmax']
        #     pcolormesh_kwargs.pop('vmax')
        # else:
        #     vmax = np.max(h) // 2
        vmax = 1

        if color_scale == 'log':
            pcm_kwargs['norm'] = LogNorm(vmax=vmax)
        elif color_scale == 'linear':
            pcm_kwargs['vmax'] = vmax
        else:
            raise ValueError('Wrong `color_scale`; should be either "log" or "linear".')

        pcm_kwargs.update(pcolormesh_kwargs)

        time_label, value_label = self.make_labels()
        
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        else:
            ax.set_xlabel(time_label)
            
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel(value_label) 

        if xlim is not None:
            ax.set_xlim(xlim)

        if ylim is not None:
            ax.set_ylim(ylim)

        cmap = copy.copy(plt.cm.__dict__[cmap])
        cmap.set_bad(cmap(0))
        pcm = ax.pcolormesh(xedges, yedges, h.T, cmap=cmap, rasterized=True, **pcm_kwargs)

        # assign colorbar to axis (instead of fig) : https://matplotlib.org/stable/gallery/subplots_axes_and_figures/colorbar_placement.html
        lb = f'{self.label} density' if self.label is not None else 'Density'
        cax = inset_axes(
            ax,
            width='3%',
            height='100%',
            loc="lower left",
            bbox_to_anchor=(1.01, 0., 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
        plt.colorbar(pcm, cax=cax, label=lb)

        if title is not None:
            ax.set_title(title, **title_kwargs)
        
        if 'fig' in locals():
            return fig, ax
        else:
            return ax
    
    def plot(self, figsize=[12, 4],
        xlabel=None, ylabel=None, title=None, ylim=None, xlim=None,
        legend_kwargs=None, title_kwargs=None, ax=None, **plot_kwargs):
        ''' Plot the raw values (multiple series)
        '''

        legend_kwargs = {} if legend_kwargs is None else legend_kwargs
        title_kwargs = {} if title_kwargs is None else title_kwargs

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        ax.margins(0)
        # plot timeseries
        _plot_kwargs = {'linewidth': 1}
        _plot_kwargs.update(plot_kwargs)

        ax.plot(self.time, self.value, **_plot_kwargs)
        
        time_label, value_label = self.make_labels()
        
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        else:
            ax.set_xlabel(time_label)
            
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel(value_label) 

        if xlim is not None:
            ax.set_xlim(xlim)

        if ylim is not None:
            ax.set_ylim(ylim)

        if title is not None:
            _title_kwargs = {'fontweight': 'bold'}
            _title_kwargs.update(title_kwargs)
            ax.set_title(title, **_title_kwargs)

        if 'fig' in locals():
            return fig, ax
        else:
            return ax
        
    def plot_traces(self, num_traces = 5, figsize=[10, 4], title=None, label = None,
                    seed = None, indices = None, xlim=None, ylim=None,
                    linestyle='-', ax=None, plot_legend=True,  lgd_kwargs=None,
                    xlabel=None, ylabel=None,  color='grey', lw=0.5, alpha=0.1):
        '''Plot EnsembleTS as a subset of traces.

        Parameters
        ----------
        
        num_traces : int
            Number of traces to plot, chosen at random. Default is 5. 

        figsize : list, optional

            The figure size. The default is [10, 4].

        xlabel : str, optional

            x-axis label. The default is None.

        ylabel : str, optional

            y-axis label. The default is None.

        title : str, optional

            Plot title. The default is None.
            
        label : str, optional
        
            Label to use on the plot legend. 
            Automatically generated if not provided. 
            
        seed : int, optional
            seed for the random number generator. Useful for reproducibility.
            The default is None. Disregarded if indices is not None
            
        indices : int, optional
            (0-based) indices of the traces. 
            The default is None. If provided, supersedes "seed" and "num_traces".

        xlim : list, optional

            x-axis limits. The default is None.

        ylim : list, optional

            y-axis limits. The default is None.

        color : str, optional

            Color of the traces. The default is sns.xkcd_rgb['pale red'].

        alpha : float, optional

            Transparency of the lines representing the multiple members. The default is 0.3.

        linestyle : {'-', '--', '-.', ':', '', (offset, on-off-seq), ...}

            Set the linestyle of the line

        lw : float, optional

            Width of the lines representing the multiple members. The default is 0.5.

        num_traces : int, optional

            Number of traces to plot. The default is None, which will plot all traces. 

        savefig_settings : dict, optional

            the dictionary of arguments for plt.savefig(); some notes below:
            - "path" must be specified; it can be any existed or non-existed path,
                with or without a suffix; if the suffix is not given in "path", it will follow "format"
            - "format" can be one of {"pdf", "eps", "png", "ps"} The default is None.

        ax : matplotlib.ax, optional

            Matplotlib axis on which to return the plot. The default is None.

        plot_legend : bool; {True,False}, optional

            Whether to plot the legend. The default is True.

        lgd_kwargs : dict, optional

            Parameters for the legend. The default is None.

        seed : int, optional

            Set the seed for the random number generator. Useful for reproducibility. The default is None.

        Returns
        -------

        fig : matplotlib.figure
        
            the figure object from matplotlib
            See [matplotlib.pyplot.figure](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.figure.html) for details.

        ax : matplotlib.axis
        
            the axis object from matplotlib
            See [matplotlib.axes](https://matplotlib.org/api/axes_api.html) for details.

            '''
        lgd_kwargs = {} if lgd_kwargs is None else lgd_kwargs.copy()
        
        time_label, value_label = self.make_labels()
        
        nts_max = 20
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            
        # Select traces to plot    
        if seed is not None:
            np.random.seed(seed)
        if indices is not None:
            num_traces = len(indices)
            if num_traces > nts_max:
                ValueError(f'Please pick a maximum of {nts_max} traces')
            trace_idx = indices
            trace_lbl = label if label is not None else f'sample paths {np.array(indices)+1}'
                
        else:
            if num_traces is not None:
                if num_traces > nts_max:
                    ValueError('num_traces is too large; reduced to '+str(nts_max))
                num_traces = np.min([num_traces,nts_max]) # cap it
                trace_idx = np.random.choice(self.nEns, num_traces, replace=False)
            else:
                trace_idx = range(nts_max)
            trace_lbl = label if label is not None else f'sample paths (n={num_traces})'
            
        # plot the traces
        for idx in trace_idx:
            ax.plot(self.time, self.value[:,idx], zorder=99, linewidth=lw,
                    color=color, alpha=alpha, linestyle='-')
        # dummy plot for trace labels
        ax.plot(np.nan, np.nan, color=color, alpha=alpha, linestyle='-',
                label=trace_lbl)

        if xlabel is not None:
            ax.set_xlabel(xlabel)
        else:
            ax.set_xlabel(time_label)
            
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel(value_label) 
            
        if title is not None:
            ax.set_title(title)
            
        if xlim is not None:
            ax.set_xlim(xlim)

        if ylim is not None:
            ax.set_xlim(ylim)

        if plot_legend:
            lgd_args = {'frameon': False}
            lgd_args.update(lgd_kwargs)
            ax.legend(**lgd_args)
            
        if 'fig' in locals():
            return fig, ax
        else:
            return ax    

    def plot_hdi(self, prob = 0.9, median=True, figsize=[12, 4], color = 'tab:blue',
        xlabel=None, ylabel=None, label=None, title=None, ylim=None, xlim=None, alpha=0.2,
        legend_kwargs=None, title_kwargs=None, ax=None, **plot_kwargs):
        '''
        h/t: Arviz code: https://arviz-devs.github.io/arviz/_modules/arviz/stats/stats.html#hdi

        Parameters
        ----------
        prob : float
            probability for which the highest density interval will be computed. The default is 0.9.
        median : bool
            If True (default), the posterior median is added.
        figsize : tuple, optional
            dimensions of the figure. The default is [12, 4].
        xlabel : str, optional
            Label for x axis. The default is None.
        ylabel : str, optional
            Label for y axis. The default is None.
        label : str, optional
            Label for the plotted objects; useful for multi-plots.
            If None (default) is specified, will attempt to use the object's label.
        title : TYPE, optional
            DESCRIPTION. The default is None.
        ylim : TYPE, optional
            DESCRIPTION. The default is None.
        xlim : TYPE, optional
            DESCRIPTION. The default is None.
        alpha : TYPE, optional
            DESCRIPTION. The default is 0.3.
        legend_kwargs : TYPE, optional
            DESCRIPTION. The default is None.
        title_kwargs : TYPE, optional
            DESCRIPTION. The default is None.
        ax : TYPE, optional
            DESCRIPTION. The default is None.
        **plot_kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''

        legend_kwargs = {} if legend_kwargs is None else legend_kwargs
        title_kwargs = {} if title_kwargs is None else title_kwargs
        
        if label is None:
            label = '' if self.label is None else self.label

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        ax.margins(0)
        
        # compute HDI
        HDI = np.zeros((self.nt,2))
        for it in range(self.nt):
            HDI[it,:] = utils.hdi1d(self.value[it,:], hdi_prob=prob)
        
        ax.fill_between(
            self.time, HDI[:,0], HDI[:,1], color=color, alpha=alpha,
            label=label+f' {prob*100}% HDI')
        
        if median == True:
            ym = self.get_median().value[:,0] 
            _plot_kwargs = {'linewidth': 1}
            _plot_kwargs.update(plot_kwargs)
            ax.plot(self.time, ym, label = label+' median', color = color, **_plot_kwargs)
        
        time_label, value_label = self.make_labels()
        
        _legend_kwargs = {'ncol': 2, 'loc': 'upper center'}
        _legend_kwargs.update(legend_kwargs)
        ax.legend(**_legend_kwargs)

        if xlabel is not None:
            ax.set_xlabel(xlabel)
        else:
            ax.set_xlabel(time_label)
            
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel(value_label) 

        if xlim is not None:
            ax.set_xlim(xlim)

        if ylim is not None:
            ax.set_ylim(ylim)

        _title_kwargs = {'fontweight': 'bold'}
        _title_kwargs.update(title_kwargs)
        
        if title is not None:       
            ax.set_title(title, **_title_kwargs)
        elif title is None and self.label is not None:
            ax.set_title(self.label, **_title_kwargs)
            
        if 'fig' in locals():
            return fig, ax
        else:
            return ax    
        

    def plot_qs(self, figsize=[10, 4], qs=[0.025, 0.25, 0.5, 0.75, 0.975], color='indianred',
        xlabel=None, ylabel=None, title=None, ylim=None, xlim=None, alphas=[0.3, 0.1],
        plot_kwargs=None, legend_kwargs=None, title_kwargs=None, ax=None, plot_trend=True):
        ''' Plot the quantiles

        Args:
            figsize (list, optional): The size of the figure. Defaults to [12, 4].
            qs (list, optional): The list to denote the quantiles plotted. Defaults to [0.025, 0.25, 0.5, 0.75, 0.975].
            color (str, optional): The basic color for the quantile envelopes. Defaults to 'indianred'.
            xlabel (str, optional): The label for the x-axis. Defaults to 'Year (CE)'.
            ylabel (str, optional): The label for the y-axis. Defaults to None.
            title (str, optional): The title of the figure. Defaults to None.
            ylim (tuple or list, optional): The limit of the y-axis. Defaults to None.
            xlim (tuple or list, optional): The limit of the x-axis. Defaults to None.
            alphas (list, optional): The alphas for the quantile envelopes. Defaults to [0.5, 0.1].
            plot_kwargs (dict, optional): The keyword arguments for the `ax.plot()` function. Defaults to None.
            legend_kwargs (dict, optional): The keyword arguments for the `ax.legend()` function. Defaults to None.
            title_kwargs (dict, optional): The keyword arguments for the `ax.title()` function. Defaults to None.
            ax (matplotlib.axes, optional): The `matplotlib.axes` object. If set the image will be plotted in the existing `ax`. Defaults to None.
            plot_trend (bool, optional): If True, will plot the trend analysis result if existed. Defaults to True.

        '''

        plot_kwargs = {} if plot_kwargs is None else plot_kwargs
        legend_kwargs = {} if legend_kwargs is None else legend_kwargs
        title_kwargs = {} if title_kwargs is None else title_kwargs

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        ax.margins(0)

        # calculate quantiles
        ts_qs = np.quantile(self.value, qs, axis=-1)
        nqs = len(qs)
        idx_mid = int(np.floor(nqs/2))

        if qs[idx_mid] == 0.5:
            label_mid = 'median'
        else:
            label_mid = f'{qs[2]*100}%'

        # plot timeseries
        _plot_kwargs = {'linewidth': 1}
        _plot_kwargs.update(plot_kwargs)

        ax.plot(self.time, ts_qs[idx_mid], label=label_mid, color=color, **_plot_kwargs)
        for i, alpha in zip(range(idx_mid), alphas[::-1]):
            ax.fill_between(
                self.time, ts_qs[-(i+1)], ts_qs[i], color=color, alpha=alpha,
                label=f'{qs[i]*100}-{qs[-(i+1)]*100}%')

        time_label, value_label = self.make_labels()
        
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        else:
            ax.set_xlabel(time_label)
            
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel(value_label) 
            
        if xlim is not None:
            ax.set_xlim(xlim)

        if ylim is not None:
            ax.set_ylim(ylim)

        _legend_kwargs = {'ncol': len(qs)//2+1, 'loc': 'upper left'}
        _legend_kwargs.update(legend_kwargs)
        leg = ax.legend(**_legend_kwargs)
        #leg.set_in_layout(False)

        _title_kwargs = {'fontweight': 'bold'}
        _title_kwargs.update(title_kwargs)
        
        if title is not None:       
            ax.set_title(title, **_title_kwargs)
        elif title is None and self.label is not None:
            ax.set_title(self.label, **_title_kwargs)

        if 'fig' in locals():
            return fig, ax
        else:
            return ax


    def plot_trend_mean_kde(self, figsize=[5, 5], ax=None, title=None, hide_ylabel=False,
                            title_kwargs=None, lgd_kwargs=None, tag='mean', label=None, color=None):
        title_kwargs = {} if title_kwargs is None else title_kwargs
        lgd_kwargs = {} if lgd_kwargs is None else lgd_kwargs
        ax.margins(0)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        xm = self.trend_dict['xm']
        xp = xm[xm>0]
        dm_kde = self.trend_dict['dm_kde']
        dm_prob = self.trend_dict['dm_prob']
        dt_kde = self.trend_dict['dt_kde']
        dt_prob = self.trend_dict['dt_prob']

        if tag == 'mean':
            kde = dm_kde
            prob = dm_prob
        elif tag == 'trend':
            kde = dt_kde
            prob = dt_prob

        nblocks, _ = np.shape(self.trend_dict['means'])
        Lb = self.nt // nblocks
        if tag == 'mean':
            if label is None:
                label = f'{Lb}y, ' + r'$P(\Delta \langle \bar{T} \rangle > 0)=$' + f'{prob:3.2f}'
            xlabel = r'$\Delta \langle \bar{T} \rangle ~({}^{\circ} C)$'
        elif tag == 'trend':
            if label is None:
                label = f'{Lb}y, ' + r'$P(\Delta \langle \dot{T} \rangle > 0)=$' + f'{prob:3.2f}'
            xlabel = r'$\Delta \langle \dot{T} \rangle ~({}^{\circ} C/\mathrm{interval})$'

        if color is not None:
            ax.fill_between(xp, kde(xp),alpha=0.3, color=color)
            ax.plot(xm, kde(xm), linewidth=2, label=label, color=color)
        else:
            ax.fill_between(xp, kde(xp),alpha=0.3)
            ax.plot(xm, kde(xm), linewidth=2, label=label)

        _lgd_kwargs = {'loc': 'upper right', 'fontsize': 12, 'bbox_to_anchor': (1.2, 1)}
        _lgd_kwargs.update(lgd_kwargs)
        ax.legend(**_lgd_kwargs)
        ax.set_xlabel(xlabel)

        if not hide_ylabel:
            ax.set_ylabel('Probability Density')

        if title is not None:
            _title_kwargs = {'fontweight': 'bold'}
            _title_kwargs.update(title_kwargs)
            ax.set_title(title, **_title_kwargs)

        if 'fig' in locals():
            return fig, ax
        else:
            return ax
        
  

            
            
            
        