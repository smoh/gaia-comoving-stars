from __future__ import division, print_function

# Third-party
import astropy.units as u
import astropy.coordinates as coord
import numpy as np

# Project
from ..gaiatools import TGASStar
from ..likelihood import get_y, get_M, get_Cinv, get_A_mu_Delta

def make_data():
    # make fake data
    n_data = 128
    ra = np.random.uniform(0, 2*np.pi, size=n_data)
    dec = np.pi/2. - np.arccos(2*np.random.uniform(size=n_data)-1.)
    parallax = np.random.uniform(-0.1, 1000., size=n_data)
    pmra,pmdec = np.random.normal(0, 100, size=(2,n_data))

    _M = np.random.uniform(0.1, 0.3, size=(n_data,5,5))
    Cov = 0.5 * np.array([M.dot(M.T) for M in _M])
    assert Cov.shape == (n_data,5,5)

    all_data = []
    for r,d,plx,pmr,pmd,C in zip(ra,dec,parallax,pmra,pmdec,Cov):
        row = {
            'ra': r, 'dec': d,
            'parallax': plx, 'pmra': pmr, 'pmdec': pmd
        }
        data = TGASStar(row)
        data._cov = C
        all_data.append(data)

    return all_data

def test_y():
    all_data = make_data()
    for data in all_data:
        d = 1000./data._parallax
        y = get_y(d, data)
        print(y.shape)
        break

# def test_M():
#     data = make_data()
#     M = get_M(data)
