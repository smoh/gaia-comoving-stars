import numpy as np
from numpy import sqrt, exp, pi, inf
from scipy import integrate



def construct_cov_pm(gaia_data):
    """
    Construct covariance matrix from gaia table for proper motion
    """
    names = ['pmra', 'pmdec']

    n = len(gaia_data['ra_error'])
    C = np.zeros((n,len(names),len(names)))

    # pre-load the diagonal
    for i,name in enumerate(names):
        full_name = "{}_error".format(name)
        C[:,i,i] = gaia_data[full_name]**2

    for i,name1 in enumerate(names):
        for j,name2 in enumerate(names):
            if j <= i: continue    
            full_name = "{}_{}_corr".format(name1, name2)
            C[...,i,j] = gaia_data[full_name]*np.sqrt(C[...,i,i]*C[...,j,j])
            C[...,j,i] = gaia_data[full_name]*np.sqrt(C[...,i,i]*C[...,j,j])
    return C


def like_distance(d, p, sigp):
    """likelihood of distance given parallax and its error
    d : distance
    p : parallax
    sigp : parallax error
    """
    return 1./sqrt(2*pi)/sigp * exp(-0.5/sigp**2 * (p-1./d)**2)

def prior_distance_uniform(d, rlim=1e3):
    """truncated uniform prior"""
    if d>0 and d<rlim:
        return 1./rlim
    return 0.

def prior_distance_constdens(d, rlim=1e3):
    """costant volume density prior on distance"""
    if d>0 and d<rlim:
        return 3./rlim**3 * d**2
    return 0.

def prior_distance_expcutoff(d, L=1e3):
    """constant density + exponential cut-off
    L : length scale"""
    if d>0:
        return 1./(2.*L**3) * d**2 * exp(-d/L)
    return 0.

def post_distance(d, p, sigp, prior=prior_distance_uniform, normalize=False):
    """posterior of ditance
    """
    if normalize:
        integrand = lambda d: like_distance(d, p, sigp)* prior(d)
        Z = integrate.quad(integrand, 0, inf)[0]
        return like_distance(d, p, sigp)*prior(d)/Z
    if prior(d)>0:
        return like_distance(d, p, sigp)*prior(d)
    return 0.


def likelihood_gaussian(X, D, C):
    # gaussian likelihood at X given Data and Covariance matrix
    # X, D : 1-d array of size ndim
    # C : 2-d array of shape ndim, ndim
    ndim = X.size
    assert X.shape == D.shape, 'X and D must have the same shape'
    assert C.shape[0] == ndim, 'covariance matrix for %i must be %i, %i' % (ndim, ndim, ndim)
    R = D-X
    invC = np.linalg.inv(C)
    return float(sqrt(np.linalg.det(invC)/ (2.*pi)**ndim) * \
                 exp(-0.5 * R[np.newaxis].dot(invC).dot(R[np.newaxis].T)))


def prior_velocity_gaussian(v_ra, v_dec, sigv=30.):
    # isotropic velocity dispersion
    return 1./(2.*pi*sigv**2) * exp(-0.5*(v_ra**2+v_dec**2)/sigv**2)


def likelihood(d1, d2, v_ra, v_dec, p1, p2, sigp1, sigp2, mu1, mu2, C1, C2):
    # mu: [pmra, pmdec]
    # v: [v_ra, v_dec]
    # d1,d2 : scalars
    v = np.array([v_ra, v_dec])
    Pmu1 = likelihood_gaussian(v/d1/4.74, mu1, C1)
    Pmu2 = likelihood_gaussian(v/d2/4.74, mu2, C2)
    Pd1 = post_distance(d1, p1, sigp1)
    Pd2 = post_distance(d2, p2, sigp2)
    Pv = prior_velocity_gaussian(v_ra, v_dec)
    return Pd1*Pd2*Pmu1*Pmu2*Pv

def likelihood_null(d1, d2, v1_ra, v1_dec, v2_ra, v2_dec, p1, p2, sigp1, sigp2, mu1, mu2, C1, C2):
    # mu: [pmra, pmdec]
    # v: [v_ra, v_dec]
    # d1,d2 : scalars
    v1 = np.array([v1_ra, v1_dec])
    Pmu1 = likelihood_gaussian(v1/d1/4.74, mu1, C1)
    v2 = np.array([v2_ra, v2_dec])
    Pmu2 = likelihood_gaussian(v2/d2/4.74, mu1, C2)
    Pd1 = post_distance(d1, p1, sigp1, prior=prior_distance_uniform)
    Pd2 = post_distance(d2, p2, sigp2, prior=prior_distance_uniform)
    Pv1 = prior_velocity_gaussian(v1_ra, v1_dec)
    Pv2 = prior_velocity_gaussian(v2_ra, v2_dec)
    return Pmu1*Pmu2*Pv1*Pv2*Pd1*Pd2
