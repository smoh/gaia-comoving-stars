from __future__ import division, print_function

# Third-party
import astropy.units as u
import numpy as np
from scipy.linalg import block_diag

# Project
from .coords import get_tangent_basis

__all__ = ['get_y', 'get_M', 'get_Cinv', 'get_Ainv_nu_Delta',
           'ln_H1_marg_v_likelihood', 'ln_Q', 'ln_H2_marg_v_likelihood']

pc_mas_yr_to_km_s = (1*u.pc * u.mas/u.yr).to(u.km/u.s,u.dimensionless_angles()).value

def get_y(ds, stars):
    ds = np.atleast_1d(ds)
    stars = np.atleast_1d(stars)

    y = np.hstack([[d * star._pmra * pc_mas_yr_to_km_s,
                    d * star._pmdec * pc_mas_yr_to_km_s,
                    star._rv] for d,star in zip(ds,stars)])
    return y

def get_M(stars):
    stars = np.atleast_1d(stars)

    M = [get_tangent_basis(star._ra, star._dec)
         for star in stars]

    return np.vstack(M)

def get_Cinv(ds, stars):
    ds = np.atleast_1d(ds)
    stars = np.atleast_1d(stars)

    Cinvs = []
    for d,star in zip(ds,stars):
        Cinv = star.get_sub_cov_inv()
        Cinv[:2,:2] /= d**2
        Cinvs.append(Cinv)

    return block_diag(*Cinvs)

def get_Ainv_nu_Delta(d, M_dirty, Cinv_dirty, y_dirty, Vinv):
    d = np.atleast_1d(d)

    # do the right thing when Cinv == 0 for RV's
    idx, = np.where(np.isclose(np.diag(Cinv_dirty), 0.))
    Cinv = np.delete(Cinv_dirty, idx, axis=0)
    Cinv = np.delete(Cinv, idx, axis=1)
    _,log_detCinv = np.linalg.slogdet(Cinv/(2*np.pi))

    M = np.delete(M_dirty, idx, axis=0)
    y = np.delete(y_dirty, idx, axis=0)

    # using ji vs. ij does the transpose of M
    Ainv = np.einsum('ji,jk,ks->is', M, Cinv, M) + Vinv

    # using ji vs. ij does the transpose
    Bb = np.einsum('ji,jk,k->i', M, Cinv, y)
    nu = -np.linalg.solve(Ainv, Bb)

    sgn,log_detVinv = np.linalg.slogdet(Vinv/(2*np.pi))

    yT_Cinv_y = np.einsum('i,ji,j->', y, Cinv, y)
    nuT_Ainv_nu = np.einsum('i,ji,j->', nu, Ainv, nu)
    Delta = (-sum([2*np.log(dd) for dd in d]) - 0.5*log_detCinv - 0.5*log_detVinv
             + 0.5*yT_Cinv_y - 0.5*nuT_Ainv_nu)

    return Ainv, nu, Delta

def _marg_likelihood_helper(ds, data, Vinv):
    y = get_y(ds, data)
    M = get_M(data)
    Cinv = get_Cinv(ds, data)

    Ainv,nu,Delta = get_Ainv_nu_Delta(ds, M, Cinv, y, Vinv)
    sgn,log_detAinv = np.linalg.slogdet(Ainv/(2*np.pi))
    log_detA = -log_detAinv
    assert sgn > 0
    return 0.5*log_detA - Delta

def ln_H1_marg_v_likelihood(d1, d2, data1, data2, Vinv):
    ds = np.array([d1, d2])
    data = [data1, data2]
    return _marg_likelihood_helper(ds, data, Vinv)

def ln_Q(d, data, Vinv):
    return _marg_likelihood_helper(d, data, Vinv)

def ln_H2_marg_v_likelihood(d1, d2, data1, data2, Vinv):
    return (ln_Q(d1, data1, Vinv) + ln_Q(d2, data2, Vinv))
