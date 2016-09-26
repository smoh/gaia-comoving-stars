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
    def ra(self):
        return self.tbl['ra']*u.degree

    @property
    def dec(self):
        return self.tbl['dec']*u.degree

    @property
    def parallax(self):
        return self.tbl['parallax']*u.mas

    @property
    def pmra(self):
        return self.tbl['pmra']*u.mas/u.yr

    @property
    def pmdec(self):
        return self.tbl['pmdec']*u.mas/u.yr

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
