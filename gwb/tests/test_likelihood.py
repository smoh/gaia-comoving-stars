from __future__ import division, print_function

# Third-party
import astropy.units as u
import astropy.coordinates as coord
import numpy as np

# Project
from ..data import TGASStar
from ..likelihood import get_y, get_M, get_Cinv, get_A_mu_Delta, ln_marg_likelihood

def make_data():
    # make fake data
    n_data = 128
    ra = np.random.uniform(0, 2*np.pi, size=n_data)
    dec = np.pi/2. - np.arccos(2*np.random.uniform(size=n_data)-1.)
    parallax = np.random.uniform(-0.1, 1000., size=n_data)
    pmra,pmdec = np.random.normal(0, 100, size=(2,n_data))

    _M = np.random.uniform(0.1, 0.3, size=(n_data,6,6))
    Cov = 0.5 * np.array([M.dot(M.T) for M in _M])
    _M[5] = 0.
    _M[:,5] = 0.
    assert Cov.shape == (n_data,6,6)

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
        assert y.shape == (1,4)
        assert np.allclose(y[:,0], 0)

def test_M():
    all_data = make_data()
    for data in all_data:
        M = get_M(data)
        assert M.shape == (1,4,3)
        assert np.allclose(np.linalg.norm(M[0], axis=1), [0, 1, 1, 1])

def test_Cinv():
    all_data = make_data()
    for data in all_data:
        d = 1000./data._parallax
        Cinv = get_Cinv(d, data)
        assert Cinv.shape == (1,4,4)
        assert np.allclose(Cinv[0], Cinv[0].T)

def test_A_mu_Delta():
    all_data = make_data()
    for data in all_data:
        d = 1000./data._parallax

        M = get_M(data)
        Cinv = get_Cinv(d, data)
        y = get_y(d, data)
        Vinv = np.diag([1/25.**2]*3)

        A, mu, Delta = get_A_mu_Delta(M, Cinv, y, Vinv)
        assert A.shape == (1,3,3)
        assert np.isfinite(A).all()

        assert mu.shape == (1,3)
        assert np.isfinite(mu).all()

        assert Delta.shape == (1,)
        assert np.isfinite(Delta).all()

def test_marg_likelihood():
    all_data = make_data()
    for data in all_data:
        d = 1000./data._parallax
        Vinv = np.diag([1/25.**2]*3)
        ll = ln_marg_likelihood(d, data, Vinv)
        assert np.isfinite(ll).all()
