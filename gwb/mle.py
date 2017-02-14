"""codes related to maximum likelihood estimation"""
import numpy as np
from numpy import pi, log, deg2rad, rad2deg
from astropy import units as u
import emcee
import corner
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
        return -np.inf

def lnprior_velocity_gaussian(vx, vy, vz, V=None):
    if V is None:
        V = np.eye(3) * 50.**2
    v = np.array([vx, vy, vz])
    invV = np.linalg.inv(V)
    return float(np.linalg.det(invV/(2.*pi)))*0.5 - 0.5* v[np.newaxis].dot(invV).dot(v[np.newaxis].T)

def lnprior_velocity_uniform(vx, vy, vz,):
    vlim = 200.
    if (abs(vx)<vlim) & (abs(vy)<vlim) & (abs(vz)<vlim):
        return 0.
    else:
        return -np.inf

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

def lnprior_sigmav(sigv):
    if sigv>10 or sigv<0:
        return -np.inf
    else:
        return 0.

def lnprob_nsamev(p, stars):
    vx, vy, vz, sigv = p[-4:]
    assert len(stars) == len(p) -4, 'boom'
    ll = []
    for d, star in zip(p[:-4], stars):
        vxc = np.random.normal(vx, sigv)
        vyc = np.random.normal(vy, sigv)
        vzc = np.random.normal(vz, sigv)
        ll.append( lnprior_distance_constdens(d)  + lnlike([d, vxc, vyc, vzc], star) )
    lpv = lnprior_velocity_uniform(vx, vy, vz)
    lpsigv = lnprior_sigmav(sigv)
    return lpv + np.sum(ll) + lpsigv

get_A = lambda star: get_tangent_basis(deg2rad(star._data['ra']), deg2rad(star._data['dec']))

class FitMCMC(object):
    ndim = 4
    lnprob = lnprob
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

        sampler = emcee.EnsembleSampler(nwalkers, FitMCMC.ndim, FitMCMC.lnprob, args=(star,))
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
        self.samples_d = np.vstack([
            1./d,
            vra/d/4.74,
            vdec/d/4.74,
            vr]).T

class FitMCMC_samev(object):
    ndim = 5
    lnprob = lnprob_samev
    def __init__(self, star1, star2, nwalkers=10, nsteps=5000, nburn=50):
        if not (isinstance(star1, TGASStar) or isinstance(star2, TGASStar)):
            raise ValueError('stars must be an instance of TGASStar class')
        self.walkers = nwalkers
        self.star1 = star1
        self.star2 = star2
        self.nsteps = nsteps
        self.nburn = nburn

        # p0 should have shape (nwalkers, ndim)
        p0 = np.vstack([
            np.random.normal(1./star1.parallax.value, (star1.parallax_error/star1.parallax**2).value, nwalkers),
            np.random.normal(1./star2.parallax.value, (star2.parallax_error/star2.parallax**2).value, nwalkers),
            np.random.normal(0, 30, nwalkers),
            np.random.normal(0, 30, nwalkers),
            np.random.normal(0, 30, nwalkers)]).T

        sampler = emcee.EnsembleSampler(nwalkers, FitMCMC_samev.ndim, FitMCMC_samev.lnprob, args=(star1, star2,))
        pos, prob, state = sampler.run_mcmc(p0, nsteps)
        samples = sampler.chain[:, nburn:, :].reshape((-1, self.ndim))
        self.sampler = sampler
        self.pos = pos
        self.prob = prob
        self.state = state
        self.samples = samples

        d1 = samples[:,0]
        d2 = samples[:,1]
        A1 = get_A(star1)
        vra1, vdec1, vr1 = A1.dot(samples[:,2:].T)
        self.samples_d1 = np.vstack([
            1./d1,
            vra1/d1/4.74,
            vdec1/d1/4.74,
            vr1]).T
        A2 = get_A(star2)
        vra2, vdec2, vr2 = A2.dot(samples[:,2:].T)
        self.samples_d2 = np.vstack([
            1./d2,
            vra2/d2/4.74,
            vdec2/d2/4.74,
            vr2]).T

class FitMCMCn_samev(object):
    ndim = None
    lnprob = lnprob_nsamev
    def __init__(self, stars, nwalkers=10, nsteps=5000, nburn=50):
        for star in stars:
            if not isinstance(star, TGASStar):
                raise ValueError('stars must be an instance of TGASStar class')
        self.walkers = nwalkers
        self.stars = stars
        self.nsteps = nsteps
        self.nburn = nburn

        # p0 should have shape (nwalkers, ndim)
        p0 = np.vstack(
            [np.random.normal(1./star.parallax.value, (star.parallax_error/star.parallax**2).value, nwalkers) for star in stars] \
            + [np.random.normal(0, 30, nwalkers), np.random.normal(0, 30, nwalkers), np.random.normal(0, 30, nwalkers), np.random.normal(0.1, 3, nwalkers)]).T
        ndim = p0.shape[1]

        sampler = emcee.EnsembleSampler(nwalkers, ndim, FitMCMCn_samev.lnprob, args=(stars,))
        out = sampler.run_mcmc(p0, nsteps)
        self.out = out
        samples = sampler.chain[:, nburn:, :].reshape((-1, ndim))
        self.sampler = sampler
        self.samples = samples

        samples_ds = []
        for i in range(len(stars)):
            d = samples[:,i]
            A1 = get_A(stars[i])
            vra, vdec, vr = A1.dot(samples[:,ndim-4:-1].T)
            samples_d = np.vstack([
                1./d,
                vra/d/4.74,
                vdec/d/4.74,
                vr ]).T
            samples_ds.append(samples_d)
        self.samples_ds = np.array(samples_ds)

    def corner(self, *args, **kwargs):
        truths = None
        labels = ['d%i' % (i) for i in range(len(self.stars))] + ['vx', 'vy', 'vz']
        return corner.corner(
            self.samples,
            truths=truths,
            labels=labels)

    def corner_d(self, i, *args, **kwargs):
        star = self.stars[i]
        truths = [star.parallax.value, star.pmra.value, star.pmdec.value, 0.]
        labels = ['p%i' % (i)] + ['pmra', 'pmdec', 'v_r']
        return corner.corner(
            self.samples_ds[i],
            truths=truths,
            labels=labels)
