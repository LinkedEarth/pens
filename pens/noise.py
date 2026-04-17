"""Noise process generators.

Adapted from the `stochastic` package (v0.6.0) by Flynn (crf204@gmail.com),
https://github.com/crflynn/stochastic. Reproduced here to remove the
`stochastic` dependency and support numpy >= 2.0.

Original MIT License notice
---------------------------
Copyright (c) Flynn <crf204@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

References
----------
Timmer, J., and M. Koenig. "On generating power law noise."
    Astronomy and Astrophysics 300 (1995): 707.
Davies, Robert B., and D. S. Harte. "Tests for Hurst effect."
    Biometrika 74, no. 1 (1987): 95-101.
Hosking, J. R. "Modeling persistence in hydrological time series using
    fractional differencing." Water Resources Research 20 (1984): 1898-1908.
"""
from functools import lru_cache

import numpy as np


class ColoredNoise:
    """Colored (power-law) noise generator.

    Generates noise whose power spectral density is proportional to
    ``(1/f)**beta``. Special cases: beta=0 white noise, beta=1 pink noise,
    beta=2 red/Brownian noise.

    Parameters
    ----------
    beta : float
        Spectral exponent.
    t : float
        Length of the time interval (used to scale frequencies).
    """

    def __init__(self, beta=0, t=1):
        if not isinstance(beta, (int, float)):
            raise TypeError("beta must be a number.")
        if not (isinstance(t, (int, float)) and t > 0):
            raise ValueError("t must be a positive number.")
        self.beta = beta
        self.t = float(t)
        self._n = None
        self._half = None
        self._frequencies = None
        self._scale = None

    def sample(self, n):
        """Generate a colored noise realization with n increments.

        Parameters
        ----------
        n : int
            Number of increments.

        Returns
        -------
        numpy.ndarray, shape (n+1,)
        """
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer.")
        n = n + 1
        if self._n != n:
            self._n = n
            self._half = (n + 1) // 2
            self._frequencies = np.fft.fftfreq(n, self.t)
            self._scale = [
                np.sqrt(0.5 * (1 / w) ** self.beta)
                for w in self._frequencies[1 : self._half]
            ]

        gn_real = np.random.normal(size=self._half - 1)
        gn_imag = np.random.normal(size=self._half - 1)
        fft = self._scale * (gn_real + 1j * gn_imag)

        if n % 2 == 0:
            f = np.concatenate(
                (
                    [0],
                    fft,
                    [
                        np.sqrt(0.5 * (1 / -self._frequencies[self._half]) ** self.beta)
                        * np.random.normal()
                    ],
                    np.conj(fft)[::-1],
                )
            )
        else:
            f = np.concatenate(([0], fft, np.conj(fft)[::-1]))

        return np.fft.ifft(f).real / np.std(f)


def _fgn_autocovariance(hurst, n):
    """Autocovariance sequence for fractional Gaussian noise."""
    ns_2h = np.arange(n + 1) ** (2 * hurst)
    return np.insert((ns_2h[:-2] - 2 * ns_2h[1:-1] + ns_2h[2:]) / 2, 0, 1)


def _fgn_dh_sqrt_eigenvals(hurst, n):
    """Square-roots of circulant matrix eigenvalues used by Davies-Harte."""
    return np.fft.irfft(_fgn_autocovariance(hurst, n))[:n] ** (1 / 2)


_cached_autocovariance = lru_cache(maxsize=8)(_fgn_autocovariance)
_cached_dh_sqrt_eigenvals = lru_cache(maxsize=8)(_fgn_dh_sqrt_eigenvals)


class FractionalGaussianNoise:
    """Fractional Gaussian noise (fGn) generator.

    Parameters
    ----------
    hurst : float
        Hurst exponent in (0, 1).  H=0.5 gives white noise.
    t : float
        Length of the time interval.
    """

    def __init__(self, hurst=0.5, t=1):
        if not isinstance(hurst, float):
            raise TypeError("Hurst value must be a float in (0, 1).")
        if not (0 < hurst < 1):
            raise ValueError("Hurst value must be in (0, 1).")
        if not (isinstance(t, (int, float)) and t > 0):
            raise ValueError("t must be a positive number.")
        self.hurst = hurst
        self.t = float(t)

    def _daviesharte(self, n):
        """Davies-Harte exact method."""
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer.")

        increment = self.t / n
        scale = increment ** self.hurst

        if self.hurst == 0.5:
            return np.random.normal(scale=scale, size=n)

        m = 2 ** (n - 2).bit_length() + 1
        sqrt_eigenvals = _cached_dh_sqrt_eigenvals(self.hurst, m)

        scale *= 2 ** (1 / 2) * (m - 1)
        w = np.random.normal(scale=scale, size=2 * m).view(complex)
        w[0] = w[0].real * 2 ** (1 / 2)
        w[-1] = w[-1].real * 2 ** (1 / 2)

        return np.fft.irfft(sqrt_eigenvals * w)[:n]

    def _hosking(self, n):
        """Hosking's exact method."""
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer.")

        increment = self.t / n
        scale = increment ** self.hurst
        gn = np.random.normal(0.0, 1.0, n)

        if self.hurst == 0.5:
            return gn * scale

        fgn = np.zeros(n)
        phi = np.zeros(n)
        psi = np.zeros(n)
        cov = _cached_autocovariance(self.hurst, n)

        fgn[0] = gn[0]
        v = 1
        phi[0] = 0

        for i in range(1, n):
            phi[i - 1] = cov[i]
            for j in range(i - 1):
                psi[j] = phi[j]
                phi[i - 1] -= psi[j] * cov[i - j - 1]
            phi[i - 1] /= v
            for j in range(i - 1):
                phi[j] = psi[j] - phi[i - 1] * psi[i - j - 2]
            v *= 1 - phi[i - 1] ** 2
            for j in range(i):
                fgn[i] += phi[j] * fgn[i - j - 1]
            fgn[i] += np.sqrt(v) * gn[i]

        return fgn * scale

    def sample(self, n, algorithm="daviesharte"):
        """Generate a fractional Gaussian noise realization.

        Parameters
        ----------
        n : int
            Number of increments.
        algorithm : {'daviesharte', 'hosking'}
            Generation algorithm.

        Returns
        -------
        numpy.ndarray, shape (n,)
        """
        if algorithm == "daviesharte":
            return self._daviesharte(n)
        elif algorithm == "hosking":
            return self._hosking(n)
        else:
            raise ValueError("algorithm must be 'daviesharte' or 'hosking'.")
