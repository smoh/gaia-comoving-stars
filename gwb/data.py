from __future__ import division, print_function

# Third-party
import astropy.coordinates as coord
import astropy.units as u
import numpy as np

__all__ = ['TGASStar']

class TGASStar(object):

    def __init__(self, tgas_row, rv=None, rv_err=None, metadata=None):
        self._row = tgas_row
        self._cov = None # for caching
        self._Cinv = None # for caching

        # radial velocities
        if rv is not None:
            if rv_err is None:
                raise ValueError("If radial velocity is provided, you must also "
                                 "provide an error.")
            self._rv = rv.to(u.km/u.s).value
            self._rv_err = rv_err.to(u.km/u.s).value
        else:
            self._rv = 0.
            self._rv_err = None

    def __getitem__(self, slc):
        return self._row[slc]

    # Astrometric data as attributes:
    @property
    def _ra(self):
        return np.radians(self['ra'])

    @property
    def ra(self):
        return (self._ra*u.radian).to(u.degree)

    @property
    def _dec(self):
        return np.radians(self['dec'])

    @property
    def dec(self):
        return (self._dec*u.radian).to(u.degree)

    @property
    def _parallax(self):
        return self['parallax'] # mas

    @property
    def parallax(self):
        return self._parallax*u.mas

    @property
    def _pmra(self):
        return self['pmra']

    @property
    def pmra(self):
        return self._pmra*u.mas/u.yr

    @property
    def _pmdec(self):
        return self['pmdec']

    @property
    def pmdec(self):
        return self._pmdec*u.mas/u.yr

    @property
    def rv(self):
        return self._rv*u.km/u.s

    # Other useful things
    def get_coord(self, with_parallax=False):
        if with_parallax:
            return coord.SkyCoord(ra=self['ra']*u.degree,
                                  dec=self['dec']*u.degree,
                                  distance=(1000./self['parallax'])*u.pc)

        else:
            return coord.SkyCoord(ra=self['ra']*u.degree,
                                  dec=self['dec']*u.degree)

    def get_cov(self):
        """
        The Gaia TGAS data table contains correlation coefficients and standard
        deviations for (ra, dec, parallax, pm_ra, pm_dec), but for most analysis
        we need covariance matrices. This converts the Gaia table into covariance
        matrix. If a radial velocity was specified on creation, this also contains
        the radial velocity variance. The base units are:
        [deg, deg, mas, mas/yr, mas/yr, km/s]
        """

        if self._cov is not None:
            return self._cov

        names = ['ra', 'dec', 'parallax', 'pmra', 'pmdec']

        C = np.zeros((6,6))

        # pre-load the diagonal
        for i,name in enumerate(names):
            full_name = "{}_error".format(name)
            C[i,i] = self._row[full_name]**2

        for i,name1 in enumerate(names):
            for j,name2 in enumerate(names):
                if j <= i:
                    continue
                full_name = "{}_{}_corr".format(name1, name2)
                C[i,j] = self[full_name] * np.sqrt(C[i,i]*C[j,j])
                C[j,i] = self[full_name] * np.sqrt(C[i,i]*C[j,j])

        if self._rv_err is not None:
            C[5,5] = self._rv_err**2

        self._cov = C
        return self._cov

    def get_distance(self, lutz_kelker=True):
        """
        TODO
        """

        if lutz_kelker:
            snr = self._parallax / self._row['parallax_error']
            if snr < 4:
                raise ValueError("S/N is smaller than 4!")
            tmp = self._parallax * (0.5 + 0.5*np.sqrt(1 - 16/snr**2))

        else:
            tmp = self._parallax

        return 1000./tmp * u.pc

