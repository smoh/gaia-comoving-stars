from __future__ import division, print_function

# Third-party
import astropy.coordinates as coord
import astropy.units as u
import numpy as np

class TGASData(object):

    def __init__(self, tbl):
        self.tbl = tbl
        self._cov = None # for caching
        self._Cinv = None # for caching

    def __getitem__(self, slc):
        return TGASData(self.tbl[slc])

    # Astrometric data as attributes:
    @property
    def _ra(self):
        return np.radians(self.tbl['ra'])

    @property
    def ra(self):
        return (self._ra*u.radian).to(u.degree)

    @property
    def _dec(self):
        return np.radians(self.tbl['dec'])

    @property
    def dec(self):
        return (self._dec*u.radian).to(u.degree)

    @property
    def _parallax(self):
        return self.tbl['parallax'] # mas

    @property
    def parallax(self):
        return self._parallax*u.mas

    @property
    def _pmra(self):
        return self.tbl['pmra']

    @property
    def pmra(self):
        return self._pmra*u.mas/u.yr

    @property
    def _pmdec(self):
        return self.tbl['pmdec']

    @property
    def pmdec(self):
        return self._pmdec*u.mas/u.yr

    @property
    def _rv(self):
        return 0. # TODO:

    @property
    def rv(self):
        return self._rv*u.km/u.s

    # Other useful things
    def get_coord(self, with_parallax=False):
        if with_parallax:
            return coord.SkyCoord(ra=self.tbl['ra']*u.degree,
                                  dec=self.tbl['dec']*u.degree,
                                  distance=(1000./self.tbl['parallax'])*u.pc)

        else:
            return coord.SkyCoord(ra=self.tbl['ra']*u.degree,
                                  dec=self.tbl['dec']*u.degree)

    def get_cov(self, units=None):
        """
        The Gaia TGAS data table contains correlation coefficients and standard
        deviations for (ra, dec, parallax, pm_ra, pm_dec), but for most analysis
        we need covariance matrices. This converts the Gaia table into covariance
        matrix.

        Parameters
        ----------
        units : list (optional)
            If None, returns the terms in the native Gaia release units
            (deg, deg, mas, mas/yr, mas/yr). If a list, must have len==5
            for each of the columns.

        """

        if self._cov is not None:
            return self._cov

        names = ['ra', 'dec', 'parallax', 'pmra', 'pmdec']

        catalog_units = [u.deg, u.deg, u.mas, u.mas/u.yr, u.mas/u.yr]
        if units is None:
            change_units = False
        else:
            change_units = True

        try:
            n = len(self.tbl['ra_error'])
        except TypeError:
            n = 1

        C = np.zeros((n,len(names),len(names)))

        # pre-load the diagonal
        for i,name in enumerate(names):
            full_name = "{}_error".format(name)
            if change_units:
                fac = (1.*catalog_units[i]**2).to(units[i]**2).value
            else:
                fac = 1.
            C[:,i,i] = self.tbl[full_name]**2 * fac

        for i,name1 in enumerate(names):
            for j,name2 in enumerate(names):
                if j <= i:
                    continue
                full_name = "{}_{}_corr".format(name1, name2)

                if change_units:
                    faci = (1*catalog_units[i]).to(units[i]).value
                    facj = (1*catalog_units[j]).to(units[j]).value
                else:
                    faci = 1.
                    facj = 1.
                C[...,i,j] = self.tbl[full_name] * np.sqrt(C[...,i,i]*C[...,j,j]) * faci * facj
                C[...,j,i] = self.tbl[full_name] * np.sqrt(C[...,i,i]*C[...,j,j]) * faci * facj

        self._cov = np.squeeze(C)
        return self._cov

    @property
    def get_Cinv(self):
        # TODO: This assumes 0's in the inverse variance for radial velocities
        #   but we might actually have some measurements (e.g., from RAVE)
        Cinv = np.zeros((4,4,len(self.tbl)))
        Cinv[:3,:3] = np.linalg.inv(self._cov)
        return Cinv
