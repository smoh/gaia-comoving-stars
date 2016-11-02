"""
    Script to compute likelihood ratios of identical star pairs
"""
from __future__ import division, print_function

__author__ = "smoh <semyeong.oh@gmail.com>"

import os
from copy import deepcopy
import logging
logger = logging.getLogger(__name__)

from astropy.io import fits
from astropy import log as logger
import numpy as np
from sklearn.neighbors import KDTree
import h5py
from schwimmbad import choose_pool

import gwb

class Worker(object):
    def __init__(self, Vinv, n_distance_samples, output_filename, prior_weights):
        self.Vinv = np.array(Vinv)
        self.prior_weights = prior_weights
        self.n_distance_samples = n_distance_samples
        self.output_filename = output_filename

    def work(self, i, star1, star2, v_scatter):
        h1 = gwb.ln_H1_FML(star1, star2, Vinv=self.Vinv, v_scatter=v_scatter,
                       n_dist_samples=self.n_distance_samples, prior_weights=self.prior_weights)
        h2 = gwb.ln_H2_FML(star1, star2, Vinv=self.Vinv, v_scatter=v_scatter,
                       n_dist_samples=self.n_distance_samples, prior_weights=self.prior_weights)
        return i, h1, h2

    def __call__(self, task):
        i, star1, star2, v_scatter = task
        return self.work(i, star1, star2, v_scatter)

    def callback(self, result):
        if result is None:
            pass

        else:
            i, h1,h2 = result
            with h5py.File(self.output_filename, 'a') as f:
                f['lnH1'][i] = h1
                f['lnH2'][i] = h2

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--tgasfile", dest="tgasfile",
                        type=str, help="Path to stacked TGAS data file.",
                        default='data/stacked_tgas.fits')
    parser.add_argument('-o', '--outname', dest='outname', required=True,
                        type=str, help="Output file name")

    parser.add_argument('--snr-cut', dest='snr_cut',
                        type=float, help='signal-to-noise ratio cut',
                        default=8.)

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--procs", dest="n_procs", default=1,
                       type=int, help="Number of processes.")
    group.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI.")

    args = parser.parse_args()


    # some hard-coded numbers
    v_scatter = 0.
    n_distance_samples = 128
    Vinvs = [np.eye(3)/15.**2,
             np.eye(3)/30.**2,
             np.eye(3)/50.**2]   # 3x3 inverse variances for gaussian prior on velocities
    prior_weights = np.array([0.3, 0.55, 0.15])
    
    snr_cut = args.snr_cut

    tgas = gwb.TGASData(args.tgasfile)
    snr = tgas.parallax_snr
    ind0 = np.arange(len(tgas))
    tgas = tgas[snr > snr_cut]
    ind0 = ind0[snr > snr_cut]
    logger.info("%i stars with S/N > %.2f" % (len(tgas), snr_cut))

    # compute maginalized likelihoods
    pool_kwargs = dict(mpi=args.mpi, processes=args.n_procs)
    pool = choose_pool(**pool_kwargs)
    logger.debug("Using pool: {}".format(pool.__class__))

    if pool.is_master():
        logger.info("Output to %s" % (args.outname))
        with h5py.File(args.outname, 'w') as f:
            f.create_dataset('ind', data=ind0)
            f.create_dataset('lnH1', shape=(ind0.size,))
            f.create_dataset('lnH2', shape=(ind0.size,))

    worker = Worker(Vinv=Vinvs, n_distance_samples=n_distance_samples,
                    output_filename=args.outname, prior_weights=prior_weights)

    allpairs = [[i, tgas[i], tgas[i], v_scatter] for i in range(len(tgas))]

    logger.info('pool start')
    for result in pool.map(worker, allpairs, callback=worker.callback):
        # returns a generator, so need to explicitly loop to do the processing, but
        #   we ignore the results because the callback function caches them.
        pass

    pool.close()
