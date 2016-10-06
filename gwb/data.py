from __future__ import division, print_function

# Third-party
from astropy.io import fits
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import six

__all__ = ['TGASData', 'TGASStar']

class TGASData(object):
    _unit_map = {
        'ra': u.degree,
        'dec': u.degree,
        'parallax': u.milliarcsecond,
        'pmra': u.milliarcsecond/u.year,
        'pmdec': u.milliarcsecond/u.year,
        'ra_error': u.degree,
        'dec_error': u.degree,
        'parallax_error': u.milliarcsecond,
        'pmra_error': u.milliarcsecond/u.year,
        'pmdec_error': u.milliarcsecond/u.year,
    }

    def __init__(self, filename_or_data, rv=None, rv_err=None):

        # radial velocities
        if rv is not None:
            if rv_err is None:
                raise ValueError("If radial velocity is provided, you must also "
                                 "provide an error.")

            if not hasattr(rv, 'unit') or not hasattr(rv_err, 'unit'):
                raise TypeError("Input radial velocity and error must be an Astropy "
                                "Quantity object.")

            elif not rv.unit.is_equivalent(u.km/u.s) or not rv_err.unit.is_equivalent(u.km/u.s):
                raise u.UnitsError("Radial velocity unit is not convertible to km/s!")

            self._rv = rv.to(u.km/u.s).value
            self._rv_err = rv_err.to(u.km/u.s).value

        else:
            self._rv = 0. # need to do this so y can have float type
            self._rv_err = None

        # TODO: maybe support memory-mapping here?
        if isinstance(filename_or_data, six.string_types):
            self._data = np.array(fits.getdata(filename_or_data, 1))

        else:
            self._data = np.array(filename_or_data)

    def __getattr__(self, name):
        # to prevent recursion errors:
        #   http://nedbatchelder.com/blog/201010/surprising_getattr_recursion.html
        if name == '_data':
            raise AttributeError()

        if name in TGASData._unit_map:
            return self._data[name] * TGASData._unit_map[name]

        elif name in self._data.dtype.names:
            return self._data[name]

        else:
            raise AttributeError("Object {} has no attribute '{}' and source data "
                                 "table has no column with that name.".format(self, name))

    def __getitem__(self, slc):
        sliced = self._data[slc]

        if self._rv_err is not None:
            rv = self._rv[slc]
            rv_err = self._rv_err[slc]
        else:
            rv = None
            rv_err = None

        if sliced.ndim == 0: # this is only one row
            return TGASStar(row=sliced, rv=rv, rv_err=rv_err)

        else: # many rows
            return TGASData(sliced, rv=rv, rv_err=rv_err)

    def __len__(self):
        return len(self._data)

    @property
    def rv(self):
        return self._rv*u.km/u.s

    # Other convenience methods
    def get_distance(self, lutz_kelker=True):
        """
        Return the distance with or without the Lutz-Kelker correction.
        """

        if lutz_kelker:
            snr = self._data['parallax'] / self._data['parallax_error']
            tmp = self._data['parallax'] * (0.5 + 0.5*np.sqrt(1 - 16/snr**2))

        else:
            tmp = self._data['parallax']

        return 1000./tmp * u.pc

    def get_vtan(self, lutz_kelker=True):
        """
        Return the tangential velocity computed using the proper motion
        and distance.
        """
        d = self.get_distance(lutz_kelker=lutz_kelker)
        vra = (self.pmra * d).to(u.km/u.s, u.dimensionless_angles()).value
        vdec = (self.pmdec * d).to(u.km/u.s, u.dimensionless_angles()).value
        return np.vstack((vra, vdec)).T * u.km/u.s

    def get_coord(self, lutz_kelker=True):
        """
        Return an `~astropy.coordinates.SkyCoord` object to represent
        all coordinates.
        """
        return coord.SkyCoord(ra=self.ra, dec=self.dec,
                              distance=self.get_distance(lutz_kelker=lutz_kelker))

    @property
    def parallax_snr(self):
        return self.parallax / self.parallax_error


class TGASStar(TGASData):

    def __init__(self, row, rv=None, rv_err=None):
        self._data = row
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
            self._rv = 0. # need to do this so y can have float type
            self._rv_err = None

    def __len__(self):
        return 1

    def __getitem__(self, slc):
        object.__getitem__(self, slc)

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
            C[i,i] = self._data[full_name]**2

        for i,name1 in enumerate(names):
            for j,name2 in enumerate(names):
                if j <= i:
                    continue
                full_name = "{}_{}_corr".format(name1, name2)
                C[i,j] = self._data[full_name] * np.sqrt(C[i,i]*C[j,j])
                C[j,i] = self._data[full_name] * np.sqrt(C[i,i]*C[j,j])

        if self._rv_err is not None:
            C[5,5] = self._rv_err**2

        self._cov = C
        return self._cov

