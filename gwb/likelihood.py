from __future__ import division, print_function

# Third-party
import astropy.coordinates as coord
import astropy.units as u
import numpy as np

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
    usph = coord.UnitSphericalRepresentation(lon=lon*u.rad, lat=lat*u.rad)
    return usph.represent_as(coord.CartesianRepresentation).xyz.value.T

def get_tangent_basis(ra, dec, dra=0.5, ddec=0.5):
    """
    column vectors are the tangent-space basis at (alpha, delta, r)
    """

    # unit vector pointing at the sky position of the target
    u_hat = get_u_vec(ra, dec)

    # unit vector offset in declination
    if dec > np.pi/4:
        v_hat = get_u_vec(ra, dec-ddec)
        dec_hat_sign = -1.
    else:
        v_hat = get_u_vec(ra, dec+ddec)
        dec_hat_sign = 1.

    dec_hat = dec_hat_sign * (v_hat - u_hat)
    ra_hat = get_u_vec(ra+dra, dec) - u_hat # always a positive offset in RA

    # define the orthogonal basis using gram-schmidt orthonormalization
    #  - u1 is the unit vector that points to (ra,dec)
    u1 = u_hat

    u2 = dec_hat - dec_hat.dot(u1)*u1
    u2 /= np.sqrt(np.sum(u2**2))

    u3 = ra_hat - ra_hat.dot(u1)*u1 - ra_hat.dot(u2)*u2
    u3 /= np.sqrt(np.sum(u3**2))

    return np.vstack((u3,u2,u1))
