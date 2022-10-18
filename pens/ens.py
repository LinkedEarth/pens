from multiprocessing.sharedctypes import Value
from matplotlib import gridspec
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import copy
import sklearn.metrics
from tqdm import tqdm
from scipy.stats import gaussian_kde 
from . import utils
import statsmodels as sm
import scipy.linalg as linalg
from scipy.stats import multivariate_normal
from tqdm import tqdm


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

    def get_mean(self):
        res = self.copy() # copy object to get metadata
        res.value = self.mean[:, np.newaxis]
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

    def get_trend(self, segment_length=10, step=10, xm=np.linspace(-0.5,1.5,200)):
        new = self.copy()
        means, trends, tm, idxs = utils.means_and_trends_ensemble(self.value, segment_length, step, self.time)
        dmeans  = means[-1:] - means[:-1] # difference of means
        dtrends = segment_length*(trends[-1:] - trends[:-1]) # difference of trends

        dm_kde = gaussian_kde(dmeans.flatten(),bw_method=0.2)
        dm_prob = dm_kde.integrate_box_1d(0, xm.max()) # estimate probability of positive change
        dt_kde = gaussian_kde(dtrends.flatten(),bw_method=0.2)
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
        n_elements = len(timespan)
        if n_elements % 2 == 1:
            raise ValueError('The number of elements in timespan must be even!')

        n_segments = int(n_elements / 2)
        mask = [False for i in range(np.size(self.time))]
        for i in range(n_segments):
            mask |= (self.time >= timespan[i*2]) & (self.time <= timespan[i*2+1])

        new = EnsembleTS(time=self.time[mask], value=self.value[mask])
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
        
        # transfer unit metadata
        es.value_name = self.value_name 
        es.value_unit = self.value_unit
        es.time_name = self.time_name
        es.value_name = self.value_name
        return es

    def sample_random(self, seed=None, n=1):
        ''' Get `n` realizations of random sample paths
        '''
        if seed is not None:
            np.random.seed(seed)

        idx = np.random.randint(low=0, high=self.nEns, size=(self.nt, n))
        path = np.ndarray((self.nt, n))
        for ie in range(n):
            for it in range(self.nt):
                path[it, ie] = self.value[it, idx[it, ie]]

        new = EnsembleTS(time=self.time, value=path)
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
        

    def compare_nearest(self, ens, metric='MSE'):
        ''' Compare with the nearest path from another EnsembleTS

        Note that we assume the size of the time axis is consistent.
        If not, please call EnsembleTS.slice() ahead.
        '''
        dist = np.zeros(ens.nEns)
        for i in tqdm(range(ens.nEns)):
            target = ens.value[:, i]
            dist[i] = self.sample_nearest(target, metric=metric).distance

        return dist

    def compare(self, ens, metric='MSE'):
        ''' Compare with another EnsembleTS

        Note that we assume the size of the time axis is consistent.
        If not, please call EnsembleTS.slice() ahead.
        '''
        dist_func = {
            'MSE': sklearn.metrics.mean_squared_error,
            'KLD': utils.kl_div,
        }
        max_nens = np.min([self.nEns, ens.nEns])
        dist = np.zeros(ens.nt)
        for i in tqdm(range(ens.nt)):
            dist[i] = dist_func[metric](self.value[i, :max_nens], ens.value[i, :max_nens])

        return dist

    def plot(self, figsize=[12, 4],
        xlabel='Year (CE)', ylabel='Value', title=None, ylim=None, xlim=None, alphas=[0.5, 0.1],
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
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

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

    def line_density(self, figsize=[12, 4], cmap='Greys', color_scale='linear', bins=None, num_fine=None,
        xlabel='Year (CE)', ylabel='Value', title=None, ylim=None, xlim=None, 
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

        fig.colorbar(pcm, ax=ax, label=self.label +' density', pad=0)

        if title is not None:
            ax.set_title(title, **title_kwargs)
        
        return fig, ax
            

    def plot_qs(self, figsize=[12, 4], qs=[0.025, 0.25, 0.5, 0.75, 0.975], color='indianred',
        xlabel=None, ylabel=None, title=None, ylim=None, xlim=None, alphas=[0.5, 0.1],
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
            plot_kwargs (dict, optional): The keyward arguments for the `ax.plot()` function. Defaults to None.
            legend_kwargs (dict, optional): The keyward arguments for the `ax.legend()` function. Defaults to None.
            title_kwargs (dict, optional): The keyward arguments for the `ax.title()` function. Defaults to None.
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
                label=f'{qs[i]*100}% to {qs[-(i+1)]*100}%')

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
            ax.set_xlim(ylim)

        _legend_kwargs = {'ncol': len(qs)//2+1, 'loc': 'upper left'}
        _legend_kwargs.update(legend_kwargs)
        ax.legend(**_legend_kwargs)

        if hasattr(self, 'trend_dict') and plot_trend:
            means = self.trend_dict['means']
            trends = self.trend_dict['trends']
            tm = self.trend_dict['tm']
            idxs = self.trend_dict['idxs']
            all_idxs = np.arange(idxs[-1,0],idxs[-1,1]+1)
            segment_years = self.time[all_idxs]
            dot = ax.scatter(tm[-1],means[-1].mean(),100,color='black',zorder=99,alpha=0.5)
            mean_line = ax.axhline(means[-1].mean(),color='black',linewidth=2,ls='-.',alpha=0.5)
            slope_segment_values = all_idxs*trends[-1].mean()
            slope_segment_values -= slope_segment_values.mean()
            slope_segment_values += means[-1].mean()
            trend_line = ax.plot(segment_years,slope_segment_values,color='black',linewidth=2)
            ax.axvspan(segment_years[0],segment_years[-1],alpha=0.2,color='silver')

        
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
        
        
    def likelihood(self, ts, phi, max_lag = 5, mute_pbar=False):
        '''
        Computes likelihood of observing a trajectory ts conditional on the ensemble.
        Assumes that deterministic trends and time-dependent scaling have been removed already.
        
        Parameters
        ----------
        
        ts : Pyleoclim Series object
            Timeseries to be evaluated against ensemble EnsTS
            
        phi : AR model parameters. phi.shape[1] sets up the model order. 
            If None, phi is estimated using statsmodels.tsa.ar_model 
            
        max_lag : int
            Maximum number of lags to take into account in AR model

        Returns
        -------
        f :  likelihood 

        '''
        y = ts.value
        mu = self.get_mean().value[:,0]
        sig = self.get_std().value[:,0]
        
        self = (self - mu)/sig
        y = (y - mu)/sig

        if len(y) != self.nt:
            raise ValueError("The series and ensemble must have the same time dimension")
             
        if phi is not None:
            mod_sel = sm.tsa.ar_model.ar_select_order(mu, maxlag=max_lag)
            ar_order = len(mod_sel.ar_lags)     
        else:
            ar_order = phi.shape[1]
            self_pyleo = self.to_pyleo()
            # estimate autocorrelation parameters
            for i, s in enumerate(self_pyleo.series_list):
                ts_mod = sm.tsa.ar_model.AutoReg(s.value, ar_order) # set up the model
                ts_res = ts_mod.fit(cov_type='HAC', cov_kwds={'maxlags': ar_order})  # Heteroskedasticity-autocorrelation robust covariance estimation.
                phi[i,:] = ts_res.params[1:] # export estimated parameters
            
        f     = np.empty((self.nEns)) # array to store likelihood of y
        fmu   = np.empty_like(f) # array to store likelihoods of the mean (for reference)
              
        for j in tqdm(range(self.nEns), total=self.nEns, disable=mute_pbar): # loop over ensemble members
            rho = np.empty((self.nt))
            x = self.value[:,j]      
            acf = sm.tsa.stattools.acf(x,nlags=ar_order)
            rho[:ar_order+1] = acf
            for k in range(ar_order+1,self.nt):
                rho[k] = phi[j,0]*rho[k-1]+phi[j,1]*rho[k-2]+phi[j,2]*rho[k-3]  # apply the recurrence relation R
            
            #rho[0] = np.var(x) # overprint first lag with the variance

            # construct the covariance matrix
            Sigma = linalg.toeplitz(rho)
            
            # compute the density
            f[j] = multivariate_normal.pdf(y,cov=Sigma, allow_singular=True)
            fmu[j] = multivariate_normal.pdf(mu,cov=Sigma,allow_singular=True)
                     
        # average them all together
        f_bar = f.mean()
        fmu_bar = fmu.mean()
        
        return f_bar, fmu_bar
            


            
            
            
        