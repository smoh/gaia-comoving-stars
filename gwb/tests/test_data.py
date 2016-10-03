from __future__ import division, print_function

# Third-party
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
import numpy as np

# Project
from ..data import TGASData, TGASStar

def test_tgasdata():
    filename = get_pkg_data_filename('data/tgas_subset.fits')

    # try both ways of initializing
    tgas = TGASData(filename)
    with fits.open(filename) as hdulist:
        tgas = TGASData(hdulist[1].data)

    assert hasattr(tgas.ra, 'unit')
    assert hasattr(tgas.ra_error, 'unit')

    assert isinstance(tgas[0:5], TGASData)
    assert isinstance(tgas[0], TGASStar)

def test_tgasstar():
    filename = get_pkg_data_filename('data/tgas_subset.fits')

    # try both ways of initializing
    tgas = TGASData(filename)
    star = tgas[0]

    assert hasattr(star.ra, 'unit')
    assert hasattr(star.ra_error, 'unit')
