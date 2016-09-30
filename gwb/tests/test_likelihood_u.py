"""

    Unit tests for gwb/likelihood.py

"""

from __future__ import division, print_function

# Third-party
import numpy as np

# Project
from ..data import TGASStar
from ..likelihood import (get_y, get_M, get_Cinv, get_Ainv_nu_Delta,
                          ln_H1_marg_v_likelihood, ln_Q, ln_H2_marg_v_likelihood)

Vinv = np.diag([1/25.**2]*3)

def make_random_data(n_batch=1, size=128):
    all_data = []
    for i in range(size):
        data = []
        for j in range(n_batch):
            # make fake data
            ra = np.random.uniform(0, 2*np.pi)
            dec = np.pi/2. - np.arccos(2*np.random.uniform()-1.)
            parallax = np.random.uniform(-0.1, 1000.)
            pmra,pmdec = np.random.normal(0, 100, size=2)

            _C = np.random.uniform(0.1, 0.3, size=(6,6))
            Cov = 0.5 * _C.dot(_C.T)
            Cov[5] = 0. # TODO: ignoring RV
            Cov[:,5] = 0. # TODO: ignoring RV
            assert Cov.shape == (6,6)

            row = {
                'ra': np.degrees(ra), 'dec': np.degrees(dec),
                'parallax': parallax, 'pmra': pmra, 'pmdec': pmdec
            }
            star = TGASStar(row)
            star._cov = Cov

            data.append(star)

        if len(data) == 1:
            all_data.append(data[0])
        else:
            all_data.append(data)

    return all_data

# ----------------------------------------------------------------------------

def test_y():
    all_data = make_random_data()
    for data in all_data:
        d = 1000./data._parallax
        y = get_y(d, data)
        assert y.shape == (3,)

    ds = [1000./data._parallax for data in all_data[:2]]
    y = get_y(ds, all_data[:2])
    assert y.shape == (6,)

def test_M():
    all_data = make_random_data()
    for data in all_data:
        M = get_M(data)
        assert M.shape == (3,3)
        assert np.allclose(np.linalg.norm(M, axis=1), [1, 1, 1])

    M = get_M(all_data[:2])
    assert M.shape == (6,3)

def test_Cinv():
    all_data = make_random_data()
    for data in all_data:
        d = 1000./data._parallax
        Cinv = get_Cinv(d, data)
        assert Cinv.shape == (3,3)
        assert np.allclose(Cinv[0], Cinv[0].T)

    ds = [1000./data._parallax for data in all_data[:2]]
    Cinv = get_Cinv(ds, all_data[:2])
    assert Cinv.shape == (6,6)
    assert np.allclose(Cinv[0], Cinv[0].T)

def test_Ainv_nu_Delta():
    all_data = make_random_data()
    for data in all_data:
        d = 1000./data._parallax

        M = get_M(data)
        Cinv = get_Cinv(d, data)
        y = get_y(d, data)

        Ainv, nu, Delta = get_Ainv_nu_Delta(d, M, Cinv, y, Vinv)
        assert Ainv.shape == (3,3)
        assert np.isfinite(Ainv).all()

        assert nu.shape == (3,)
        assert np.isfinite(nu).all()

        assert np.isfinite(Delta)

    ds = [1000./data._parallax for data in all_data[:2]]
    M = get_M(all_data[:2])
    Cinv = get_Cinv(ds, all_data[:2])
    y = get_y(ds, all_data[:2])
    Ainv, nu, Delta = get_Ainv_nu_Delta(ds, M, Cinv, y, Vinv)
    assert Ainv.shape == (3,3)
    assert np.isfinite(Ainv).all()
    assert nu.shape == (3,)
    assert np.isfinite(nu).all()
    assert np.isfinite(Delta)

def test_H1_marg():
    for star1,star2 in make_random_data(n_batch=2):
        d1 = 1000/star1._parallax
        d2 = 1000/star1._parallax
        ll = ln_H1_marg_v_likelihood(d1, d2, star1, star2, Vinv)
        assert np.isfinite(ll)

def test_H2_marg():
    for star1,star2 in make_random_data(n_batch=2):
        d1 = 1000/star1._parallax
        d2 = 1000/star1._parallax
        ll = ln_H2_marg_v_likelihood(d1, d2, star1, star2, Vinv)
        assert np.isfinite(ll)
