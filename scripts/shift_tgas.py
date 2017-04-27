from __future__ import division, print_function

import os

from astropy.table import Table
from astropy import log as logger
import numpy as np

def shift_tgas(tgas_path, outname, dra=0., ddec=0., dpmra=0., dpmdec=0.):
    """
    Shift all of tgas star position and proper motions by specified values

    dra, ddec : float,
        in degrees
    dpmra, dpmdec : float,
        in mas/yr
    """
    tgas = Table.read(tgas_path)
    columns = ['parallax', 'pmra', 'pmdec', 'parallax_error', 'pmra_error', 'pmdec_error',
               'parallax_pmra_corr', 'parallax_pmdec_corr', 'pmra_pmdec_corr',
               'ra', 'dec', 'ra_error', 'dec_error',
               'ra_dec_corr', 'ra_parallax_corr', 'ra_pmra_corr', 'ra_pmdec_corr',
               'dec_parallax_corr', 'dec_pmra_corr', 'dec_pmdec_corr']
    logger.info("dra = %.2f ddec = %.2f dpmra = %.2f dpmdec = %.2f" % (
                dra, ddec, dpmra, dpmdec))
    tgas['ra'] += dra
    tgas['dec'] += ddec
    tgas['pmra'] += dpmra
    tgas['pmdec'] += dpmdec
    logger.info("writing the new tgas to %s" % (outname))
    tgas[columns].write(outname, overwrite=False)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('tgas_path', type=str)
    parser.add_argument('outname', type=str)
    parser.add_argument('--dra', type=float, default=0.)
    parser.add_argument('--ddec', type=float, default=0.)
    parser.add_argument('--dpmra', type=float, default=0.)
    parser.add_argument('--dpmdec', type=float, default=0.)
    args = parser.parse_args()
    shift_tgas(args.tgas_path, args.outname,
               dra=args.dra, ddec=args.ddec, dpmra=args.dpmra, dpmdec=args.dpmdec)


