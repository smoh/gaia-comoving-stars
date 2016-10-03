from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os

# Third-party
from astropy.io import fits
from astropy import log as logger
import numpy as np
from sklearn.neighbors import KDTree

# Project
from gwb.data import TGASData

def main(stacked_tgas_path, signal_to_noise_cut, n_neighbors, delta_v_cut,
         output_path="../data/", overwrite=False):

    if not os.path.exists(os.path.abspath(stacked_tgas_path)):
        raise IOError("Stacked TGAS data file '{}' does not exist.".format(stacked_tgas_path))

    if not os.path.exists(os.path.abspath(output_path)):
        raise IOError("Specified output path '{}' does not exist.".format(output_path))

    output_file = os.path.join(output_path, "snr{:.0f}_n{}_dv{:.0f}.fits".format(signal_to_noise_cut,
                                                                                 n_neighbors,
                                                                                 delta_v_cut))
    if os.path.exists(output_file) and not overwrite:
        raise IOError("Output file '{}' already exists. Use --overwrite to overwrite.")

    tgas = TGASData(os.path.abspath(stacked_tgas_path))
    n_full_tgas = len(tgas)

    # first, do a signal-to-noise cut
    tgas = tgas[tgas.parallax_snr > signal_to_noise_cut]
    logger.info("{}/{} targets left after S/N cut".format(len(tgas), n_full_tgas))

    # next, built the KD Tree using the XYZ positions
    c = tgas.get_coord()
    X = c.cartesian.xyz.T

    # separation in position
    tree = KDTree(X)
    tree_d,tree_i = tree.query(X, k=n_neighbors+1) # 0th match is always self
    tree_d = tree_d[:,1:]
    tree_i = tree_i[:,1:]

    idx0 = np.arange(len(tgas), dtype=int)
    idx0 = np.repeat(idx0[:,None], n_neighbors, axis=1)
    all_pair_idx = np.vstack((idx0.ravel(), tree_i.ravel())).T

    # now compute velocity difference
    vtan = tgas.get_vtan().value
    dv = np.sqrt(np.sum((vtan[all_pair_idx[:,0]] - vtan[all_pair_idx[:,1]])**2, axis=1))
    cut = dv < delta_v_cut

    all_pair_idx = all_pair_idx[cut]
    dv = dv[cut]
    sep = tree_d.ravel()[cut]
    logger.info("{} pairs before trimming duplicates".format(len(all_pair_idx)))

    hitting_edge = np.bincount(all_pair_idx[:,0]) == n_neighbors
    logger.info("{} stars likely have more than {} neighbors".format(hitting_edge.sum(),
                                                                     n_neighbors))

    all_pair_idx = np.sort(all_pair_idx, axis=1)
    str_pairs = np.array(["{}{}".format(i,j) for i,j in all_pair_idx])
    _, unq_idx = np.unique(str_pairs, return_index=True)
    all_pair_idx = all_pair_idx[unq_idx]
    dv = dv[unq_idx]
    sep = sep[unq_idx]
    logger.info("{} pairs after trimming duplicates".format(len(all_pair_idx)))

    rows = [(i1,i2,x,y) for i1,i2,x,y in zip(all_pair_idx[:,0], all_pair_idx[:,1], dv, sep)]
    tbl = np.array(rows, dtype=[('star1', 'i8'), ('star2', 'i8'),
                                ('delta_v', 'f8'), ('sep', 'f8')])

    hdu = fits.BinTableHDU(tbl)
    hdu.writeto(output_file, clobber=True)

if __name__ == "__main__":
    from argparse import ArgumentParser
    import logging

    # Define parser object
    parser = ArgumentParser(description="")

    vq_group = parser.add_mutually_exclusive_group()
    vq_group.add_argument('-v', '--verbose', action='count', default=0, dest='verbosity')
    vq_group.add_argument('-q', '--quiet', action='count', default=0, dest='quietness')

    parser.add_argument('-s', '--seed', dest='seed', default=None,
                        type=int, help='Random number generator seed.')
    parser.add_argument('-o', '--overwrite', action='store_true', dest='overwrite',
                        default=False, help='Destroy everything.')

    parser.add_argument("--tgas-file", dest="stacked_tgas_path", required=True,
                        type=str, help="Path to stacked TGAS data file.")
    parser.add_argument("--snr-cut", dest="signal_to_noise_cut", default=8,
                        type=float, help="Minimum signal-to-noise ratio in parallax.")
    parser.add_argument("--neighbors", dest="n_neighbors", default=32,
                        type=int, help="Number of nearest neighbors to process for each star.")
    parser.add_argument("--deltav-cut", dest="delta_v_cut", default=4,
                        type=float, help="TODO.")
    parser.add_argument("--output-path", dest="output_path", default="../data/",
                        type=str, help="Path to write output.")

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbosity != 0:
        if args.verbosity == 1:
            logger.setLevel(logging.DEBUG)
        else: # anything >= 2
            logger.setLevel(1)

    elif args.quietness != 0:
        if args.quietness == 1:
            logger.setLevel(logging.WARNING)
        else: # anything >= 2
            logger.setLevel(logging.ERROR)

    else: # default
        logger.setLevel(logging.INFO)

    if args.seed is not None:
        np.random.seed(args.seed)

    main(args.stacked_tgas_path, args.signal_to_noise_cut, args.n_neighbors, args.delta_v_cut,
         output_path=args.output_path, overwrite=args.overwrite)
