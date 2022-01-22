from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import copy
from . import utils

class EnsembleTS:
    ''' Ensemble Timeseries

    Note that annual reconstruction is assumed so the time axis is in years.
    The ensembles variable should be in shape of (nt, nEns), where nt is the number of years,
    and nEns is the number of ensemble members.

    '''
    def __init__(self, time=None, value=None):
        self.time = time
        self.value = value

        if self.value is not None:
            self.nt = np.shape(self.value)[0]
            self.nEns = np.shape(self.value)[-1]
            self.median = np.median(self.value, axis=-1)

    def copy(self):
        return copy.deepcopy(self)
    
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

    def sample_path(self, seed=None, n=1):
        if seed is not None:
            np.random.seed(seed)

        idx = np.random.randint(low=0, high=self.nEns, size=(self.nt, n))
        path = np.ndarray((self.nt, n))
        for ie in range(n):
            for it in range(self.nt):
                path[it, ie] = self.value[it, idx[it, ie]]

        new = EnsembleTS(time=self.time, value=path)
        return new


    def plot(self, figsize=[12, 4],
        xlabel='Year (CE)', ylabel='Value', title=None, ylim=None, xlim=None, alphas=[0.5, 0.1],
        plot_kwargs=None, legend_kwargs=None, title_kwargs=None, ax=None):
        ''' Plot the raw values
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
            ax.set_xlim(ylim)

        if title is not None:
            _title_kwargs = {'fontweight': 'bold'}
            _title_kwargs.update(title_kwargs)
            ax.set_title(title, **_title_kwargs)

        if 'fig' in locals():
            return fig, ax
        else:
            return ax
            

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
