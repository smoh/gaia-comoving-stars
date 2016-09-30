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

def get_tangent_basis(ra, dec, dra=0.5, ddec=0.5):
    """
    row vectors are the tangent-space basis at (alpha, delta, r)
    """

    ra = np.atleast_1d(ra)
    dec = np.atleast_1d(dec)

    # unit vector pointing at the sky position of the target
    u_hat = get_u_vec(ra, dec)

    # unit vector offset in declination
    dec_hat_sign = np.ones_like(u_hat)
    dec_hat_sign[:,dec > np.pi/4] = -1.

    v_hat = np.zeros_like(u_hat)
    v_hat[:,dec <= np.pi/4] = get_u_vec(ra[dec <= np.pi/4], dec[dec <= np.pi/4]+ddec)
    v_hat[:,dec > np.pi/4] = get_u_vec(ra[dec > np.pi/4], dec[dec > np.pi/4]-ddec)

    dec_hat = dec_hat_sign * (v_hat - u_hat)
    ra_hat = get_u_vec(ra+dra, dec) - u_hat # always a positive offset in RA

    # define the orthogonal basis using gram-schmidt orthonormalization
    #  - u1 is the unit vector that points to (ra,dec)
    u1 = u_hat

    u2 = dec_hat - np.einsum('ij,ij->j', dec_hat, u1)*u1
    u2 /= np.sqrt(np.sum(u2**2, axis=0))

    u3 = (ra_hat - np.einsum('ij,ij->j', ra_hat, u1)*u1 -
          np.einsum('ij,ij->j', ra_hat, u2)*u2)
    u3 /= np.sqrt(np.sum(u3**2, axis=0))

    T = np.stack((u3,u2,u1))
    return np.squeeze(np.moveaxis(T,-1,0))
