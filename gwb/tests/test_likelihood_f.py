"""

    Functional tests for gwb/likelihood.py

"""

from __future__ import division, print_function

# Third-party
import numpy as np

# Project
from ..data import TGASStar
from ..coords import get_tangent_basis
from ..likelihood import (get_y_Cinv, get_M, get_Ainv_nu_Delta,
                          ln_H1_marg_v_likelihood, ln_Q, ln_H2_marg_v_likelihood)

Vinv = np.diag([1/25.**2]*3)

def make_data_increase_uncertainties(n_data=128):
    ra = np.full(n_data, np.random.uniform(0, 2*np.pi))
    dec = np.full(n_data, np.pi/2. - np.arccos(2*np.random.uniform()-1.))
    parallax = np.ones(n_data)*10. # 100 pc
    pmra = 25/4.740470463496208 * parallax
    pmdec = -25/4.740470463496208 * parallax

    all_data = []
    for i,a in enumerate(np.linspace(-8,-1,n_data)):
        Cov = np.diag(np.full(6, 2**a))**2
        row = {
            'ra': np.degrees(ra[i]), 'dec': np.degrees(dec[i]),
            'parallax': parallax[i], 'pmra': pmra[i], 'pmdec': pmdec[i]
        }
        data = TGASStar(row)
        data._cov = Cov
        all_data.append(data)

    return all_data

# ----------------------------------------------------------------------------

def test_M():
    # make sure the projection matrix M is the same for all of these data
    all_data = make_data_increase_uncertainties()
    comp_M = None
    for data in all_data:
        M = get_M(data)
        if comp_M is None:
            comp_M = M
        assert np.allclose(M, comp_M)

def test_Cinv():
    # as uncertainties increase, determinant of inverse variance matrix should get smaller
    all_data = make_data_increase_uncertainties()
    all_det_Cinv = []
    for data in all_data:
        d = 1000./data._data['parallax']
        y,Cinv = get_y_Cinv(d, data)
        sgn,det = np.linalg.slogdet(Cinv[:2,:2])
        all_det_Cinv.append(det)

    all_det_Cinv = np.array(all_det_Cinv)
    assert np.all((all_det_Cinv[1:] - all_det_Cinv[:-1]) < 0)

def test_Ainv_nu_Delta():
    # as uncertainties increase, nu^T A^-1 nu should *increase*
    all_data = make_data_increase_uncertainties()
    all_nuT_Ainv_nu = []
    all_log_detA = []
    for data in all_data:
        d = 1000./data._data['parallax']
        M = get_M(data)
        y,Cinv = get_y_Cinv(d, data)

        Ainv, nu, Delta = get_Ainv_nu_Delta(d, M, Cinv, y, Vinv)
        all_log_detA.append(-np.linalg.slogdet(Ainv)[1])
        all_nuT_Ainv_nu.append(nu.dot(Ainv).dot(nu))

    # A should increase as uncertainties increase
    all_log_detA = np.array(all_log_detA)
    assert np.all((all_log_detA[1:] - all_log_detA[:-1]) > 0)

    # nuT_Ainv_nu goes like Sigma^-1, so should decrease
    all_nuT_Ainv_nu = np.array(all_nuT_Ainv_nu)
    assert np.all((all_nuT_Ainv_nu[1:] - all_nuT_Ainv_nu[:-1]) < 0)

# def test_plot_shape_of_Q():
#     Vinv = np.diag([1/25.**2]*3)
#     for pair in make_good_pairs():
#         d1 = 1000/pair[0]._parallax
#         print(d1, pair[0]._parallax)
#         d_grid = np.linspace(0.1, 8*d1, 256)

#         ll = np.zeros_like(d_grid)
#         for i,d in enumerate(d_grid):
#             ll[i] = ln_Q(d, pair[0], Vinv)

#         import matplotlib.pyplot as plt
#         plt.plot(d_grid, np.exp(ll-ll.max()))
#         plt.show()

#         return
