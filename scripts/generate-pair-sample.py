"""

    To generate random pairs, run with:

        python generate-pair-sample.py random --size=10000

    To generate pairs from N nearest neighbors, run with:

        python generate-pair-sample.py neighbors -n 32 --dv 8

"""

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

def check_paths(stacked_tgas_path, output_path, output_file, overwrite):
    if not os.path.exists(os.path.abspath(stacked_tgas_path)):
        raise IOError("Stacked TGAS data file '{}' does not exist.".format(stacked_tgas_path))

    if not os.path.exists(os.path.abspath(output_path)):
        raise IOError("Specified output path '{}' does not exist.".format(output_path))

    if os.path.exists(output_file) and not overwrite:
        raise IOError("Output file '{}' already exists. Use --overwrite to overwrite.")

def get_tgas(stacked_tgas_path, signal_to_noise_cut):
    # the full TGAS data table
    tgas0 = TGASData(os.path.abspath(stacked_tgas_path))
    n_full_tgas = len(tgas0)
    index0 = np.arange(n_full_tgas)

    # do a signal-to-noise cut and preserve the indices of the surviving targets
    #   from the original stacked_tgas file
    tgas = tgas0[tgas0.parallax_snr > signal_to_noise_cut]  # this should return a copy of tgas0
    index0_snr = index0[tgas0.parallax_snr > signal_to_noise_cut]
    logger.info("{}/{} targets left after S/N cut".format(len(tgas), n_full_tgas))

    # convert the sky position and Lutz-Kelker corrected distance into a 3D cartesian position
    c = tgas.get_coord()
    X = c.cartesian.xyz.T

    return index0_snr, tgas, X

def main_random(stacked_tgas_path, output_path, signal_to_noise_cut, size, overwrite=False):

    output_file = os.path.join(output_path, "snr{:.0f}_random{:d}.fits"
                               .format(signal_to_noise_cut, size))
    check_paths(stacked_tgas_path, output_path, output_file, overwrite)

    index0_snr, tgas, X = get_tgas(stacked_tgas_path, signal_to_noise_cut)

    all_pair_idx = np.random.randint(0, len(tgas), size=(size, 2))
    sep = np.linalg.norm(X[all_pair_idx[:,0]] - X[all_pair_idx[:,1]], axis=1)
    vtan = tgas.get_vtan().value
    dv = np.sqrt(np.sum((vtan[all_pair_idx[:,0]] - vtan[all_pair_idx[:,1]])**2, axis=1))
    index0_out = np.vstack([index0_snr[all_pair_idx[:,0]], index0_snr[all_pair_idx[:,1]]]).T

    rows = [(i1,i2,x,y) for i1,i2,x,y in zip(index0_out[:,0], index0_out[:,1], dv, sep)]
    tbl = np.array(rows, dtype=[('star1', 'i8'), ('star2', 'i8'),
                                ('delta_v', 'f8'), ('sep', 'f8')])

    return output_file, tbl

def main_neighbors(stacked_tgas_path, output_path, signal_to_noise_cut, n_neighbors, delta_v_cut,
                   overwrite=False):

    output_file = os.path.join(output_path, "snr{:.0f}_n{}_dv{:.0f}.fits"
                               .format(signal_to_noise_cut, n_neighbors, delta_v_cut))
    check_paths(stacked_tgas_path, output_path, output_file, overwrite)

    index0_snr, tgas, X = get_tgas(stacked_tgas_path, signal_to_noise_cut)

    # next, built the KD Tree using the XYZ positions
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
    index0_out = np.vstack([index0_snr[all_pair_idx[:,0]], index0_snr[all_pair_idx[:,1]]]).T
    dv = dv[unq_idx]
    sep = sep[unq_idx]
    logger.info("{} pairs after trimming duplicates".format(len(all_pair_idx)))

    rows = [(i1,i2,x,y) for i1,i2,x,y in zip(index0_out[:,0], index0_out[:,1], dv, sep)]
    tbl = np.array(rows, dtype=[('star1', 'i8'), ('star2', 'i8'),
                                ('delta_v', 'f8'), ('sep', 'f8')])

    return output_file, tbl

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

    parser.add_argument("--tgas-file", dest="stacked_tgas_path", default="../data/stacked_tgas.fits",
                        type=str, help="Path to stacked TGAS data file.")
    parser.add_argument("--snr-cut", dest="signal_to_noise_cut", default=8,
                        type=float, help="Minimum signal-to-noise ratio in parallax.")
    parser.add_argument("--output-path", dest="output_path", default="../data/",
                        type=str, help="Path to write output.")

    subparsers = parser.add_subparsers()

    # parser for the "random" command
    parser_a = subparsers.add_parser('random', help='Generate indices for random pairs of stars.')
    parser_a.set_defaults(case='random')
    parser_a.add_argument("--size", dest="size", help="Number of random pairs to generate",
                          type=int, default=10000)

    # parser for the "neighbors" command
    parser_b = subparsers.add_parser('neighbors', help='Find indices for pairs of nearest neighbors')
    parser_b.set_defaults(case='neighbors')
    parser_b.add_argument("-n", "--neighbors", dest="n_neighbors", default=32,
                          type=int, help="Number of nearest neighbors to process for each star.")
    parser_b.add_argument("--dv", "--deltav-cut", dest="delta_v_cut", default=8,
                          type=float, help="perpendicular velocity cut in km/s")

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

    # --

    if args.case == 'random':
        output_file,tbl = main_random(stacked_tgas_path=args.stacked_tgas_path,
                                      output_path=args.output_path,
                                      signal_to_noise_cut=args.signal_to_noise_cut,
                                      size=args.size, overwrite=args.overwrite)

    elif args.case == 'neighbors':
        output_file,tbl = main_neighbors(stacked_tgas_path=args.stacked_tgas_path,
                                         output_path=args.output_path,
                                         signal_to_noise_cut=args.signal_to_noise_cut,
                                         n_neighbors=args.n_neighbors,
                                         delta_v_cut=args.delta_v_cut,
                                         overwrite=args.overwrite)

    required_names = ['star1', 'star2', 'delta_v', 'sep']
    for name in required_names:
        assert name in tbl.dtype.names

    hdu = fits.BinTableHDU(tbl)
    hdu.writeto(output_file, clobber=True)
