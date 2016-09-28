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
    return u_hat / np.sqrt(np.sum(u_hat**2, axis=0))[None]

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

    u3 = ra_hat - np.einsum('ij,ij->j', ra_hat, u1)*u1 - np.einsum('ij,ij->j', ra_hat, u2)*u2
    u3 /= np.sqrt(np.sum(u3**2, axis=0))

    return np.squeeze(np.stack((u3,u2,u1), axis=1))

def marg_likelihood_helper(v, d, data):
    """
    Compute the log-marginal-likelihood for Hypothesis 2: each star in a
    pair has *different* 3-space velocities. This is equivalent to log(Q)
    using the notation from the paper.

    Parameters
    ----------
    v : array_like [km/s]
        3-space velocity vector - assumes this is [v^perp_ra, v^perp_dec, v_r]
        (but that distinction is irrelevant for a spherical isotropic velocity prior).
    d : numeric [pc]
        Distance in parsecs.
    data

    """
    ra, dec, x, Cinv = data
    A = get_tangent_basis(ra, dec)
    v = A.dot(v)

    x_th = np.array([1000./d, v[0]/d*km_spc_to_mas_yr, v[1]/d*km_spc_to_mas_yr, v[2]])

    dx = x-x_th
    chi2 = np.einsum('i,ji,j->', dx, Cinv, dx)

    # determinant of the non-zero parts
    sgn,logdet = np.linalg.slogdet(Cinv[:3,:3]/(2*np.pi))

    return -0.5*chi2 + 0.5*logdet

def ln_H1_marg_likelihood(v, d1, d2, data1, data2):
    """
    Compute the log-marginal-likelihood for Hypothesis 1: each star in a
    pair has the same 3-space velocity.

    Parameters
    ----------
    v : array_like [km/s]
        3-space velocity vector - assumes this is [v^perp_ra, v^perp_dec, v_r]
        (but that distinction is irrelevant for a spherical isotropic velocity prior).
    d : numeric [pc]
        Distance in parsecs.
    x : array_like
        len-4 vector of (parallax, pmra, pmdec, RV)
    Cinv : array_like
        array of shape (4,4) for covariances

    """
    # ra1, dec1, x1, Cinv1 = data1
    # A1 = get_tangent_basis(ra1, dec1)

    # ra2, dec2, x2, Cinv2 = data2
    # A2 = get_tangent_basis(ra2, dec2)

    # # project v onto tangent basis for each star
    # v1 = A1.dot(v)
    # v2 = A2.dot(v)

    return marg_likelihood_helper(v, d1, data1) + marg_likelihood_helper(v, d2, data2)

def ln_H2_marg_likelihood(v1, v2, d1, d2, data1, data2):
    """
    Compute the log-marginal-likelihood for Hypothesis 2: each star in a
    pair has *different* 3-space velocities. This is equivalent to log(Q)
    using the notation from the paper.

    Parameters
    ----------
    v : array_like [km/s]
        3-space velocity vector - assumes this is [v^perp_ra, v^perp_dec, v_r]
        (but that distinction is irrelevant for a spherical isotropic velocity prior).
    d : numeric [pc]
        Distance in parsecs.
    x : array_like
        len-4 vector of (parallax, pmra, pmdec, RV)
    Cinv : array_like
        array of shape (4,4) for covariances

    """
    return marg_likelihood_helper(v1, d1, data1) + marg_likelihood_helper(v2, d2, data2)
