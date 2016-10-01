from __future__ import division, print_function

# Third-party
import astropy.units as u
import numpy as np

km_spc_to_mas_yr = (1*u.km/u.s/u.pc).to(u.mas/u.yr,u.dimensionless_angles()).value

def get_u_vec(lon, lat):
    """
    Given two sky coordinates as a longitude and latitude (RA, Dec),
    return a unit vector that points in the direction of the sky position.

    Sky positions should be in radians!

    Parameters
    ----------
    lon : numeric [rad]
        Longitude in radians.
    lat : numeric [rad]
        Latitude in radians.

    Returns
    -------
    u_hat : `numpy.ndarray`
        Unit 3-vector.

    """
    u_hat = np.array([np.cos(lon) * np.cos(lat),
                      np.sin(lon) * np.cos(lat),
                      np.sin(lat)])
    return u_hat

def get_tangent_basis(ra, dec):
    """
    row vectors are the tangent-space basis at (alpha, delta, r)
    """
    ra = np.atleast_1d(ra)
    dec = np.atleast_1d(dec)
    s = []
    for cra, cdec in zip(ra, dec):
        s.append(
            np.array([
                [-np.sin(cra),  np.cos(cra), 0.],
                [-np.sin(cdec)*np.cos(cra), -np.sin(cdec)*np.sin(cra), np.cos(cdec)],
                [np.cos(cdec)*np.cos(cra),  np.cos(cdec)*np.sin(cra), np.sin(cdec)]
                ]))
    return np.array(s).squeeze()
