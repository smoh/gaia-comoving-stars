"""

"""

from __future__ import division, print_function

# Third-party
import numpy as np

# Project
from ..data import TGASStar
from ..coords import get_tangent_basis
from ..fml import ln_H1_FML, ln_H2_FML

Vinv = np.diag([1/25.**2]*3)

def make_bad_pairs():
    n_data = 128

    for i in range(n_data):
        ra1,ra2 = np.random.uniform(0, 2*np.pi, size=2)
        dec1,dec2 = np.pi/2. - np.arccos(2*np.random.uniform(size=2)-1.)
        plx1,plx2 = np.exp(np.random.uniform(0,2,size=2))

        true_v1 = np.random.normal(0, 25, size=3)
        T1 = get_tangent_basis(ra1, dec1)
        v1 = T1.dot(true_v1)
        pmra1,pmdec1 = v1[:2] * plx1 / 4.740470463496208
        # TODO: ignoring RV for now

        true_v2 = np.random.normal(0, 25, size=3)
        T2 = get_tangent_basis(ra2, dec2)
        v2 = T2.dot(true_v2)
        pmra2,pmdec2 = v2[:2] * plx2 / 4.740470463496208
        # TODO: ignoring RV for now

        print(ra1,dec1,pmra1, pmdec1, plx1)
        print(ra2,dec2,pmra2, pmdec2, plx2)
        print(true_v1)
        print(true_v2)

        Cov = np.diag([1E-4]*6)**2
        # Cov = np.diag(np.random.uniform(0.1, 0.3, size=6))**2
        Cov[5] = 0. # TODO: ignoring RV for now
        Cov[:,5] = 0. # TODO: ignoring RV for now

        # HACK: they have the same covariance matrix
        star1 = TGASStar({'ra': np.degrees(ra1), 'dec': np.degrees(dec1),
                          'parallax': plx1, 'pmra': pmra1, 'pmdec': pmdec1})
        star1._cov = Cov

        star2 = TGASStar({'ra': np.degrees(ra2), 'dec': np.degrees(dec2),
                          'parallax': plx2, 'pmra': pmra2, 'pmdec': pmdec2})
        star2._cov = Cov

        yield [star1, star2]

def make_good_pairs():
    n_data = 128

    for i in range(n_data):
        ra1 = np.random.uniform(0, 2*np.pi)
        dec1 = np.pi/2. - np.arccos(2*np.random.uniform()-1.)
        plx1 = 10 ** np.random.uniform(0,2)

        ra2 = ra1 + np.random.uniform(-0.02, 0.02)
        if dec1 > 45:
            dec2 = dec1 - np.random.uniform(0, 0.03)
        else:
            dec2 = dec1 + np.random.uniform(0, 0.03)

        plx2 = plx1 + np.random.uniform(-0.01, 0.01)

        true_v = np.random.normal(0, 25, size=3)

        T1 = get_tangent_basis(ra1, dec1)
        v1 = T1.dot(true_v)
        pmra1,pmdec1 = v1[:2] * plx1 / 4.740470463496208
        # TODO: ignoring RV for now

        T2 = get_tangent_basis(ra2, dec2)
        v2 = T2.dot(true_v)
        pmra2,pmdec2 = v2[:2] * plx2 / 4.740470463496208
        # TODO: ignoring RV for now

        Cov = np.diag(np.random.uniform(0.1, 0.3, size=6))**2
        # Cov = np.diag(np.full(6,1E-2))**2
        Cov[5] = 0. # TODO: ignoring RV for now
        Cov[:,5] = 0. # TODO: ignoring RV for now

        # HACK: they have the same covariance matrix
        star1 = TGASStar({'ra': np.degrees(ra1), 'dec': np.degrees(dec1),
                          'parallax': plx1, 'pmra': pmra1, 'pmdec': pmdec1})
        star1._cov = Cov

        star2 = TGASStar({'ra': np.degrees(ra2), 'dec': np.degrees(dec2),
                          'parallax': plx2, 'pmra': pmra2, 'pmdec': pmdec2})
        star2._cov = Cov

        yield [star1, star2]

def _compute_ratio(pair):
    ll_H1 = ln_H1_FML(pair[0], pair[1], Vinv)
    ll_H2 = ln_H2_FML(pair[0], pair[1], Vinv)
    return ll_H1-ll_H2

def test_ln_likelihood_ratio():
    np.random.seed(12345)
    for pair in make_bad_pairs():
        assert _compute_ratio(pair) < 0
        print(_compute_ratio(pair))

    # for pair in make_good_pairs():
    #     assert _compute_ratio(pair) > -10
    #     break
