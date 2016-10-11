"""codes related to maximul likelihood esimation"""
import numpy as np
from numpy import pi, log, deg2rad, rad2deg
from astropy import units as u
import emcee
from .data import TGASStar
from .coords import get_tangent_basis

def lnlikelihood_gaussian(X, D, C):
    # Returns gaussian ln(likelihood) at X given Data and Covariance matrix
    # X, D : 1-d array of size ndim
    # C : 2-d array of shape ndim, ndim
    ndim = X.size
    assert X.shape == D.shape, 'X and D must have the same shape'
    assert C.shape[0] == ndim, 'covariance matrix for %i must be %i, %i' % (ndim, ndim, ndim)
    R = D-X
    invC = np.linalg.inv(C)
    return float(np.linalg.det(invC/(2.*pi)))*0.5 - 0.5* R[np.newaxis].dot(invC).dot(R[np.newaxis].T)

def lnprior_distance_constdens(d, rlim=10.):
    """constant volume density prior on distance"""
    if d>0 and d<rlim:
        return -2.*log(d)
    else:
        return -inf

def lnprior_velocity_gaussian(vx, vy, vz, V=None):
    if V is None:
        V = np.eye(3) * 50.**2
    v = np.array([vx, vy, vz])
    invV = np.linalg.inv(V)
    return float(np.linalg.det(invV/(2.*pi)))*0.5 - 0.5* v[np.newaxis].dot(invV).dot(v[np.newaxis].T)

def lnlike(p, star):
    d, vx, vy, vz = p
    ra, dec = star.ra.to(u.rad).value, star.dec.to(u.rad).value
    p, pmra, pmdec = star._data['parallax'], star._data['pmra'], star._data['pmdec']
    C = star.get_cov()[2:5,2:5]
    A = np.eye(4)  # transformation (1/d,vx,vy,vz) to (p,vra,vdec,vr)
    A[1:,1:] = get_tangent_basis(ra, dec)/d/4.74
    Xm = A.dot([1./d, vx, vy, vz])[:3]  # throw away RV
    D = np.array([p, pmra, pmdec])
    return lnlikelihood_gaussian(Xm, D, C)

def lnprob(p, star):
    d, vx, vy, vz = p
    ll = lnlike(p, star)
    lp = lnprior_distance_constdens(d) + lnprior_velocity_gaussian(vx, vy, vz)
    return ll+lp

def lnprob_samev(p, star1, star2):
    d1, d2, vx, vy, vz = p
    ll1 = lnlike(np.array([d1,vx,vy,vz]), star1)
    ll2 = lnlike(np.array([d2,vx,vy,vz]), star2)
    lp = lnprior_distance_constdens(d1) + lnprior_distance_constdens(d2) + lnprior_velocity_gaussian(vx, vy, vz)
    return ll1+ll2+lp

get_A = lambda star: get_tangent_basis(deg2rad(star._data['ra']), deg2rad(star._data['dec']))

class FitMCMC(object):
    ndim = 4
    # lnprob = lnprob
    def __init__(self, star, nwalkers=10, nsteps=5000, nburn=50):
        if not isinstance(star, TGASStar):
            raise ValueError('star must be an instance of TGASStar class')
        self.nwalkers = nwalkers
        self.star = star
        self.nsteps = nsteps
        self.nburn = nburn

        # p0 should have shape (nwalkers, ndim)
        p0 = np.vstack([
            np.random.normal(1./star.parallax.value, (star.parallax_error/star.parallax**2).value, nwalkers),
            np.random.normal(0, 30, nwalkers),
            np.random.normal(0, 30, nwalkers),
            np.random.normal(0, 30, nwalkers)]).T

        sampler = emcee.EnsembleSampler(nwalkers, self.ndim, lnprob, args=(star,))
        pos, prob, state = sampler.run_mcmc(p0, nsteps)
        samples = sampler.chain[:, nburn:, :].reshape((-1, self.ndim))
        self.sampler = sampler
        self.pos = pos
        self.prob = prob
        self.state = state
        self.samples = samples

        A = get_A(star)
        d = samples[:,0]
        vra, vdec, vr = A.dot(samples[:,1:].T)
        self.samples_vradecr = np.vstack([
            1./d,
            vra/d/4.74,
            vdec/d/4.74,
            vr]).T