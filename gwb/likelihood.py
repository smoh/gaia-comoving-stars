from __future__ import division, print_function

# Third-party
import astropy.units as u
import numpy as np
from scipy.linalg import block_diag

# Project
from .coords import get_tangent_basis

pc_mas_yr_to_km_s = (1*u.pc * u.mas/u.yr).to(u.km/u.s,u.dimensionless_angles()).value

def get_y(ds, stars):
    ds = np.atleast_1d(ds)
    stars = np.atleast_1d(stars)

    y = np.hstack([[d * star._parallax*1E-3 - 1.,
                    d * star._pmra * pc_mas_yr_to_km_s,
                    d * star._pmdec * pc_mas_yr_to_km_s,
                    star._rv] for d,star in zip(ds,stars)])
    return y

def get_M(stars):
    stars = np.atleast_1d(stars)
    M0 = np.zeros(3)

    M = [np.vstack((M0,get_tangent_basis(star._ra, star._dec)))
         for star in stars]

    return np.vstack(M)

def get_Cinv(ds, stars):
    ds = np.atleast_1d(ds)
    stars = np.atleast_1d(stars)

    Cinvs = []
    for d,star in zip(ds,stars):
        Cinv = star.get_sub_cov_inv()
        Cinv[:3,:3] /= d**2
        Cinvs.append(Cinv)

    return block_diag(*Cinvs)

def get_A_nu_Delta(d, M, Cinv, y, Vinv):
    assert Cinv.ndim == 2

    d = np.atleast_1d(d)

    # using ji vs. ij does the transpose of M
    # Ainv = np.einsum('ji,jk,ks->is', M, Cinv, M) + Vinv
    Ainv = (M.T @ (Cinv @ M)) + Vinv
    A = np.linalg.inv(Ainv)

    # using ji vs. ij does the transpose
    # Bb = np.einsum('ji,jk,k->i', M, Cinv, y)
    # nu = np.einsum('ij,j->i', A, Bb)
    Bb = M.T @ (Cinv @ y)
    nu = A @ Bb

    # do the right thing when Cinv == 0 for RV's
    idx, = np.where(np.isclose(np.diag(Cinv), 0.))
    Cinv_clean = np.delete(Cinv, idx, axis=0)
    Cinv_clean = np.delete(Cinv_clean, idx, axis=1)
    _,log_detCinv = np.linalg.slogdet(Cinv_clean/(2*np.pi))

    sgn,log_detVinv = np.linalg.slogdet(Vinv/(2*np.pi))

    yT_Cinv_y = np.einsum('i,ji,j->', y, Cinv, y)
    nuT_Ainv_nu = np.einsum('i,ji,j->', nu, Ainv, nu)
    Delta = (sum([-3*np.log(dd) for dd in d]) - 0.5*log_detCinv - 0.5*log_detVinv +
             0.5*yT_Cinv_y - 0.5*nuT_Ainv_nu)

    return A, nu, Delta

def ln_H1_marg_likelihood(d1, d2, data1, data2, Vinv):
    ds = np.array([d1, d2])
    data = [data1, data2]

    y = get_y(ds, data)
    M = get_M(data)
    Cinv = get_Cinv(ds, data)

    A,nu,Delta = get_A_nu_Delta(ds, M, Cinv, y, Vinv)
    sgn,log_detA = np.linalg.slogdet(2*np.pi*A)
    assert sgn > 0
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
            ln_H2_marg_likelihood_helper(d2, data2, Vinv))
