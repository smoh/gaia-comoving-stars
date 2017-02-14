"""
    Script to compute likelihood ratios of random samples by velocity shuffling
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
            i,h1,h2 = result
            with h5py.File(self.output_filename, 'a') as f:
                f['lnH1'][i] = h1
                f['lnH2'][i] = h2

def prepare(tgasfile, outname, snr_cut, radius_cut, radius_cut_vshuf, dvtan_cut):
    """
    """
    tgas0 = gwb.TGASData(tgasfile)
    ind0 = np.arange(len(tgas))
    tgas = tgas0[tgas0.parallax_snr > snr_cut]
    ind0 = ind0[tgas0.parallax_snr > snr_cut]
    logger.info("%i stars with S/N > %.2f" % (len(tgas), snr_cut))
    coords = tgas.get_coord()
    xyz  = coords.cartesian.xyz.T.value
    vtan = tgas.get_vtan().value

    tree = KDTree(xyz)

    Ntarget = 100000
    batchsize = 100000
    
    pairi = np.random.randint(high=len(tgas), size=Ntarget)
    pairjk = []
    for i  in pairi:
        treei = tree.query_radius([xyz[i]], 100)[0]
        j = np.random.choice(treei)
        k = np.random.choice(treei)
        pairjk.append((j,k))

        
        


    treei = tree.query_radius(xyz, radius_cut)
    sizes = np.array([x.size for x in treei])
    logger.info("total number of pairs for r<%.1f = %i" % (radius_cut, sizes.sum()))
    shufi = tree.query_radius(xyz, radius_cut_vshuf)  # instead of deepcopy, do it again
    list(map(np.random.shuffle, shufi))               # and shuffle
    pairijk = np.array([[i, j, k] for i in np.arange(len(treei)) for j, k in zip(treei[i],shufi[i])])

    sep = np.linalg.norm(xyz[pairijk[:,0]] - xyz[pairijk[:,1]], axis=1)
    dvtan = np.linalg.norm(vtan[pairijk[:,0]] - vtan[pairijk[:,2]], axis=1)
    cond = dvtan < dvtan_cut
    logger.info("total number of pairs with dvtan<%.1f = %i" % (dvtan_cut, cond.sum()))
    # if cond.sum() > npairs_limit:
    #     logger.info("choosing %i pairs randomly out of %i" % (npairs_limit, cond.sum()))
    #     #TODO
    pairijk = pairijk[cond]
    ind0_ijk = ind0[pairijk]
    sep = sep[cond]
    dvtan = dvtan[cond]
    Npairs = pairijk.shape[0]

    # record indices and aux info
    # yikes, is this safe with mpi?
    logger.info("writing pair indices in %s" % (outname))
    if not os.path.exists(outname):
        with h5py.File(outname, 'w') as f:
            f.create_dataset('ijk', data=ind0_ijk)
            f.create_dataset('sep', data=sep)
            f.create_dataset('dvtan', data=dvtan)
            f.create_dataset('lnH1', dtype='f8', shape=(Npairs,))
            f.create_dataset('lnH2', dtype='f8', shape=(Npairs,))
    return tgas, pairijk

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
    parser.add_argument('--radius-cut', dest='radius_cut',
                        type=float, help='radius cut [pc]',
                        default=10)
    parser.add_argument('--radius-cut-vshuf', dest='radius_cut_vshuf',
                        type=float, help='radius cut for velocity shuffling [pc]',
                        default=10)
    parser.add_argument('--dvtan-cut', dest='dvtan_cut',
                        type=float, help='dvtan cut [km/s]',
                        default=10)

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



    tgas, pairijk = prepare(args.tgasfile, args.outname, args.snr_cut, args.radius_cut,
                            args.radius_cut_vshuf, args.dvtan_cut)

    # compute maginalized likelihoods
    pool_kwargs = dict(mpi=args.mpi, processes=args.n_procs)
    pool = choose_pool(**pool_kwargs)
    logger.debug("Using pool: {}".format(pool.__class__))

    worker = Worker(Vinv=Vinvs, n_distance_samples=n_distance_samples,
                    output_filename=args.outname, prior_weights=prior_weights)

    # make fake stars with ra, dec, parallax from star_j and
    # proper motions from star_k for each star_i
    def make_fake_star(starj, stark):
        row = starj._data.copy()
        row['pmra'] = stark._data['pmra']
        row['pmdec'] = stark._data['pmdec']
        return gwb.TGASStar(row)
    allpairs = [[irow, tgas[i], make_fake_star(tgas[j],tgas[k]), v_scatter] for irow, (i,j,k) in enumerate(pairijk)]

    logger.info('pool start')
    for result in pool.map(worker, allpairs, callback=worker.callback):
        # returns a generator, so need to explicitly loop to do the processing, but
        #   we ignore the results because the callback function caches them.
        pass

    pool.close()
