from __future__ import division, print_function

# Third-party
import astropy.units as u
import numpy as np

# Project
from .coords import get_tangent_basis

pc_mas_yr_to_km_s = (1*u.pc * u.mas/u.yr).to(u.km/u.s,u.dimensionless_angles()).value

def get_y(d, star):
    d = np.atleast_1d(d)
    y = np.stack([d * star._parallax*1E-3 - 1.,
                  d * star._pmra * pc_mas_yr_to_km_s,
                  d * star._pmdec * pc_mas_yr_to_km_s,
                  np.atleast_1d(star._rv)], axis=1)
    return y

def get_M(star):
    ra = np.atleast_1d(star._ra)
    dec = np.atleast_1d(star._dec)
    M0 = np.zeros((3,) + ra.shape)
    M_vel = get_tangent_basis(ra, dec)
    return np.hstack((M0, M_vel)).T[None]

def get_Cinv(d, star):
    Cinv = star.get_sub_cov_inv()
    Cinv[:3,:3] /= d**2
    return Cinv[None]

def get_A_nu_Delta(d, M, Cinv, y, Vinv):
    if Cinv.ndim > 2:
        n = Cinv.shape[0]
    else:
        n = 1

    # using ji vs. ij does the transpose of M
    Ainv = np.einsum('...ji,...jk,...ks->...is', M, Cinv, M) + Vinv
    A = np.linalg.inv(Ainv)

    # using ji vs. ij does the transpose
    Bb = -np.einsum('...ji,...jk,...k->...i', M, Cinv, y)
    nu = np.einsum('...ij,...j->...i', A, Bb)

    # do the right thing when Cinv[3,3] == 0
    idx = np.isclose(Cinv[...,3,3], 0)

    log_detCinv = np.zeros(n)
    _,log_detCinv[idx] = np.linalg.slogdet(Cinv[idx,:3,:3]/(2*np.pi))
    _,log_detCinv[~idx] = np.linalg.slogdet(Cinv[~idx]/(2*np.pi))

    sgn,log_detVinv = np.linalg.slogdet(Vinv/(2*np.pi))

    yT_Cinv_y = np.einsum('...i,...ji,...j->...', y, Cinv, y)
    nuT_Ainv_nu = np.einsum('...i,...ji,...j->...', nu, Ainv, nu)
    Delta = -3*np.log(d) - 0.5*log_detCinv - 0.5*log_detVinv + 0.5*yT_Cinv_y - 0.5*nuT_Ainv_nu

    return A, nu, Delta

def ln_H1_marg_likelihood(d1, d2, data1, data2, Vinv):
    d = np.array([d1, d2])

    y1 = get_y(d1, data1)
    y2 = get_y(d2, data2)
    y = np.vstack((y1, y2))

    M1 = get_M(data1)
    M2 = get_M(data2)
    M = np.vstack((M1,M2))

    Cinv1 = get_Cinv(d1, data1)
    Cinv2 = get_Cinv(d2, data2)
    Cinv = np.vstack((Cinv1,Cinv2))

    A,nu,Delta = get_A_nu_Delta(d, M, Cinv, y, Vinv)
    _,log_detA = np.linalg.slogdet(2*np.pi*A)
    return (0.5*log_detA - Delta).sum()

def ln_H2_marg_likelihood_helper(d, data, Vinv):
    y = get_y(d, data)
    M = get_M(data)
    Cinv = get_Cinv(d, data)
    A,nu,Delta = get_A_nu_Delta(d, M, Cinv, y, Vinv)
    _,log_detA = np.linalg.slogdet(2*np.pi*A)
    return 0.5*log_detA - Delta

def ln_H2_marg_likelihood(d1, d2, data1, data2, Vinv):
    return (ln_H2_marg_likelihood_helper(d1, data1, Vinv) +
            ln_H2_marg_likelihood_helper(d2, data2, Vinv))[0]
