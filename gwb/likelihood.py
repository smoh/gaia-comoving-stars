from __future__ import division, print_function

# Third-party
import astropy.units as u
import numpy as np
from scipy.linalg import block_diag

# Project
from .coords import get_tangent_basis

__all__ = ['get_y_Cinv', 'get_M', 'get_Ainv_nu_Delta',
           'ln_H1_marg_v_likelihood', 'ln_Q', 'ln_H2_marg_v_likelihood']

pc_mas_yr_per_km_s = (1*u.km/u.s).to(u.pc*u.mas/u.yr, u.dimensionless_angles()).value
km_s_per_pc_mas_yr = 1/pc_mas_yr_per_km_s

def get_y_Cinv(ds, stars, v_scatter=0.):
    """
    Construct the vector y, which should have length 3*n_stars
    """
    ds = np.atleast_1d(ds)
    stars = np.atleast_1d(stars)

    y = np.hstack([[d * star._pmra * km_s_per_pc_mas_yr,
                    d * star._pmdec * km_s_per_pc_mas_yr,
                    star._rv] for d,star in zip(ds,stars)])

    # Construct the matrix Cinv, which should have shape (3*n_stars, 3*n_stars)
    ds = np.atleast_1d(ds)
    stars = np.atleast_1d(stars)
    assert len(ds) == len(stars)

    Cinvs = []
    for d,star in zip(ds,stars):
        Cov = star.get_cov().copy() # returns full 6x6 covariance matrix from Gaia data
        Cov = Cov[3:,3:]

        Cov[0,:] *= d * km_s_per_pc_mas_yr # pmra
        Cov[1,:] *= d * km_s_per_pc_mas_yr # pmdec
        Cov[:,0] *= d * km_s_per_pc_mas_yr # pmra
        Cov[:,1] *= d * km_s_per_pc_mas_yr # pmdec
        Cov += np.eye(Cov.shape[0]) * v_scatter**2

        Cinv = np.zeros((3,3))
        if star._rv_err is None: # no radial velocity
            Cinv[:2,:2] = np.linalg.inv(Cov[:-1,:-1])
        else:
            Cinv = np.linalg.inv(Cov)

        Cinvs.append(Cinv)

    return y, block_diag(*Cinvs)

def get_M(stars):
    """
    Construct the matrix M, which should have shape (3*n_stars, 3*n_stars)
    """
    stars = np.atleast_1d(stars)

    M = [get_tangent_basis(np.radians(star._ra), np.radians(star._dec))
         for star in stars]

    return np.vstack(M)

def get_Ainv_nu_Delta(d, M_dirty, Cinv_dirty, y_dirty, Vinv):
    """

    Issues
    ------
    - The inconsistency of units between elements of y_dirty and Cinv_dirty
      could lead to ill-conditioned matrices!

    Parameters
    ----------
    d : numeric [pc]
        Distance.
    M_dirty : array_like
        Transformation matrix.
    Cinv_dirty : array_like [1/(km/s)^2]
    y_dirty : array_like [km/s]
    Vinv : array_like
        1/(km/s)^2

    """
    d = np.atleast_1d(d)

    # If inverse variances of RV is zero (RV info unavailable),
    # delete covariances with RV
    idx = np.arange(2, Cinv_dirty.shape[-1], 3) # TODO: this is bad because 3 is hardset
    noRV = Cinv_dirty[idx,idx] == 0.
    Cinv = np.delete(Cinv_dirty, idx[noRV], axis=0)
    Cinv = np.delete(Cinv, idx[noRV], axis=1)
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

def _marg_likelihood_helper(ds, data, Vinv, v_scatter):
    y,Cinv = get_y_Cinv(ds, data, v_scatter)
    M = get_M(data)

    Ainv,nu,Delta = get_Ainv_nu_Delta(ds, M, Cinv, y, Vinv)
    sgn,log_detAinv = np.linalg.slogdet(Ainv/(2*np.pi))
    log_detA = -log_detAinv

    if sgn <= 0:
        print("y",y)
        print("M",M)
        print("Cinv",Cinv,np.linalg.eigvalsh(Cinv))
        print("Vinv",Vinv,np.linalg.eigvalsh(Vinv))
        print("Ainv",Ainv,np.linalg.eigvalsh(Ainv))

    assert sgn > 0
    return 0.5*log_detA - Delta

def ln_H1_marg_v_likelihood(d1, d2, data1, data2, Vinv, v_scatter=0.):
    ds = np.array([d1, d2])
    data = [data1, data2]
    return _marg_likelihood_helper(ds, data, Vinv, v_scatter)

def ln_Q(d, data, Vinv, v_scatter=0.):
    return _marg_likelihood_helper(d, data, Vinv, v_scatter)

def ln_H2_marg_v_likelihood(d1, d2, data1, data2, Vinv, v_scatter=0.):
    return (ln_Q(d1, data1, Vinv, v_scatter) + ln_Q(d2, data2, Vinv, v_scatter))
