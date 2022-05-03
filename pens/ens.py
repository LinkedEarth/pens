from multiprocessing.sharedctypes import Value
from matplotlib import gridspec
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import copy
from sklearn.metrics import mean_squared_error as mse
from tqdm import tqdm
from . import utils

class EnsembleTS:
    ''' Ensemble Timeseries

    Note that annual reconstruction is assumed so the time axis is in years.
    The ensembles variable should be in shape of (nt, nEns), where nt is the number of years,
    and nEns is the number of ensemble members.

    '''
    def __init__(self, time=None, value=None):
        if np.ndim(value) == 1:
            value = value[:, np.newaxis]

        self.time = time
        self.value = value

        if self.value is not None:
            self.nt = np.shape(self.value)[0]
            self.nEns = np.shape(self.value)[-1]
            self.median = np.median(self.value, axis=-1)

    def copy(self):
        return copy.deepcopy(self)

    def get_median(self):
        med = EnsembleTS(time=self.time, value=self.median)
        return med

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
        time = np.array([t.year for t in ds[time_name].values])
        nt = len(time)
        value = np.reshape(ds[var].values, (nt, -1))

        new = EnsembleTS(time=time, value=value)
        return new

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
            raise ImportError('Need to install Pyleoclim: pip install "pens[pyleo]"')

        series_list = []
        for i in range(self.nEns):
            ts = pyleo.Series(time=self.time, value=self.value[..., i], **kwargs)
            series_list.append(ts)

        if len(series_list) == 1:
            es = ts
        else:
            es = pyleo.EnsembleSeries(series_list)    
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
            'MSE': mse,
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
        

    def compare(self, ens, metric='MSE'):
        ''' Compare with another EnsembleTS

        Note that we assume the size of the time axis is consistent.
        If not, please call EnsembleTS.slice() ahead.
        '''
        dist = np.zeros(ens.nEns)
        for i in tqdm(range(ens.nEns)):
            target = ens.value[:, i]
            dist[i] = self.sample_nearest(target, metric=metric).distance

        return dist

    def plot(self, figsize=[12, 4],
        xlabel='Year (CE)', ylabel='Value', title=None, ylim=None, xlim=None, alphas=[0.5, 0.1],
        plot_kwargs=None, legend_kwargs=None, title_kwargs=None, ax=None):
        ''' Plot the raw values (multiple series)
        '''

        plot_kwargs = {} if plot_kwargs is None else plot_kwargs
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

    def line_density(self, figsize=[12, 4], cmap='plasma', color_scale='linear', bins=None, num_fine=None,
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
        - https://matplotlib.org/3.5.0/gallery/statistics/time_series_histogram.html

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

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if xlim is not None:
            ax.set_xlim(xlim)

        if ylim is not None:
            ax.set_ylim(ylim)

        cmap = copy.copy(plt.cm.__dict__[cmap])
        cmap.set_bad(cmap(0))
        pcm = ax.pcolormesh(xedges, yedges, h.T, cmap=cmap, rasterized=True, **pcm_kwargs)

        fig.colorbar(pcm, ax=ax, label='Density', pad=0)

        if title is not None:
            ax.set_title(title, **title_kwargs)
        
        return fig, ax
            

    def plot_qs(self, figsize=[12, 4], qs=[0.025, 0.25, 0.5, 0.75, 0.975], color='indianred',
        xlabel='Year (CE)', ylabel='Value', title=None, ylim=None, xlim=None, alphas=[0.5, 0.1],
        plot_kwargs=None, legend_kwargs=None, title_kwargs=None, ax=None):
        ''' Plot the quantiles
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
            ax.fill_between(self.time, ts_qs[-(i+1)], ts_qs[i], color=color, alpha=alpha, label=f'{qs[i]*100}% to {qs[-(i+1)]*100}%')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if xlim is not None:
            ax.set_xlim(xlim)

        if ylim is not None:
            ax.set_xlim(ylim)

        _legend_kwargs = {'ncol': 3, 'loc': 'upper left'}
        _legend_kwargs.update(legend_kwargs)
        ax.legend(**_legend_kwargs)


        if title is not None:
            _title_kwargs = {'fontweight': 'bold'}
            _title_kwargs.update(title_kwargs)
            ax.set_title(title, **_title_kwargs)

        if 'fig' in locals():
            return fig, ax
        else:
            return ax
