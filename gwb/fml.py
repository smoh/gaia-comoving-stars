from __future__ import division, print_function

# Third-party
import numpy as np
from scipy.misc import logsumexp

# Project
from .likelihood import ln_H1_marg_v_likelihood, ln_H2_marg_v_likelihood

__all__ = ['ln_H1_FML', 'ln_H2_FML']

def ln_parallax_prior(parallax, min_parallax=0.1):
    """
    Here we assume a uniform space density.

    Parameters
    ----------
    parallax : numeric, array_like [mas]
        The parallax(es) to evaluate the prior at.
    min_parallax : numeric (optional)
        The minimum parallax (maximum distance) to consider.
        The default is 0.1, a distance of 10 kpc.

    Returns
    -------
    lnp : `numpy.ndarray`
        An array of log-prior values evaluated at the input parallaxes.
    """
    parallax = np.atleast_1d(parallax)
    lnp = np.full(parallax.shape, -np.inf)
    good_ix = parallax > min_parallax
    lnp[good_ix] = -4*np.log(parallax[good_ix])
    return lnp

def get_posterior_distance_samples(star, size=1, min_parallax=0.1):
    """

    Parameters
    ----------
    star : `gwb.TGASStar`
    size : int (optional)
        The number of samples to return.
    min_parallax : numeric (optional)
        Passed through to ``ln_parallax_prior()``.
        The minimum parallax (maximum distance) to consider.
        The default is 0.1, a distance of 10 kpc.

    Returns
    -------
    distances : `numpy.ndarray`
        An array of samples from the distance posterior pdf for
        the given star. Will have shape ``(size,)``.
    """
    batch_size = size # MAGIC NUMBER
    maxiter = 8192 # MAGIC NUMBER

    samples = np.array([])
    iter = 0
    while len(samples) < size or iter == maxiter:
        batch = np.random.normal(star._data['parallax'], np.sqrt(star.get_cov()[0,0]), size=batch_size)
        tmps = ln_parallax_prior(batch, min_parallax=min_parallax)
        tmps -= tmps.max()

        uu_idx = np.log(np.random.uniform(size=batch_size)) < tmps
        samples = np.concatenate((samples, batch[uu_idx]))

        iter += 1

    if iter == maxiter:
        raise ValueError("Sampling reached maximum number of iterations.")

    return 1000/samples[:size]

def ln_H1_FML(star1, star2, Vinv, n_dist_samples=128, v_scatter=0., prior_weights=None):
    dist1 = get_posterior_distance_samples(star1, size=n_dist_samples)
    dist2 = get_posterior_distance_samples(star2, size=n_dist_samples)
    ll_H1_at_samples = np.array([ln_H1_marg_v_likelihood(d1, d2, star1, star2, Vinv, v_scatter, prior_weights=prior_weights)
                                 for d1,d2 in zip(dist1, dist2)])
    return logsumexp(ll_H1_at_samples) - np.log(float(ll_H1_at_samples.size))

def ln_H2_FML(star1, star2, Vinv, n_dist_samples=128, v_scatter=0., prior_weights=None):
    dist1 = get_posterior_distance_samples(star1, size=n_dist_samples)
    dist2 = get_posterior_distance_samples(star2, size=n_dist_samples)
    ll_H2_at_samples = np.array([ln_H2_marg_v_likelihood(d1, d2, star1, star2, Vinv, v_scatter, prior_weights=prior_weights)
                                 for d1,d2 in zip(dist1, dist2)])
    return logsumexp(ll_H2_at_samples) - np.log(float(ll_H2_at_samples.size))
