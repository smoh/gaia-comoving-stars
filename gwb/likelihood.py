from __future__ import division, print_function

# Third-party
import astropy.units as u
import numpy as np

# Project
from .coords import get_tangent_basis

pc_mas_yr_to_km_s = (1*u.pc * u.mas/u.yr).to(u.km/u.s,u.dimensionless_angles()).value

def get_y(d, data):
    d = np.atleast_1d(d)
    y = np.stack([d * data._parallax*1E-3 - 1.,
                  d * data._pmra * pc_mas_yr_to_km_s,
                  d * data._pmdec * pc_mas_yr_to_km_s,
                  np.atleast_1d(data._rv)], axis=1)
    return y

def get_M(data):
    M0 = np.zeros((3,) + data._ra.shape)
    M_vel = get_tangent_basis(data._ra, data._dec)
    return np.stack((M0[None], M_vel), axis=1)

def get_Cinv(d, data):
    Cinv = data.get_Cinv()
    return Cinv[:3,:3] / d**2

def get_A_mu_Delta(M, Cinv, y, Vinv):
    # TODO: is A is actually the inverse of what you think it is?
    Ainv = np.einsum('...ij,...jk,...ks->is', M.T, Cinv, M) + Vinv
    A = np.linalg.inv(Ainv)

    Bb = np.einsum('...ij,...jk,...k->i', M.T, Cinv, y)
    mu = np.einsum('...ij,...j->i', A, Bb)

    sgn,log_detCinv = np.slogdet(Cinv/(2*np.pi)) # TODO: do the right thing when Cinv[3,3] == 0
    sgn,log_detVinv = np.slogdet(Vinv/(2*np.pi))
    yT_Cinv_y = np.einsum('...i,...ji,...j->', y, Cinv, y)
    muT_A_mu = np.einsum('...i,...ji,...j->', mu, A, mu)
    Delta = 0.5*log_detCinv + 0.5*log_detVinv - 0.5*yT_Cinv_y - muT_A_mu

    return A, mu, Delta

def ln_marg_likelihood(d, data, Vinv):
    y = get_y(d, data)
    M = get_M(data)
    Cinv = get_Cinv(d, data)
    A,mu,Delta = get_A_mu_Delta(M, Cinv, y, Vinv)
    _,log_Adet = np.linalg.slogdet(A/(2*np.pi))
    return -Delta + 0.5*log_Adet # TODO: plus or minus? is A actually Ainv?
