from __future__ import division, print_function

# Third-party
import numpy as np

# Project
from ..data import TGASStar
from ..coords import get_tangent_basis
from ..likelihood import (get_y, get_M, get_Cinv, get_A_nu_Delta,
                          ln_H1_marg_likelihood, ln_H2_marg_likelihood_helper,
                          ln_H2_marg_likelihood)

def make_random_data():
    # make fake data
    n_data = 128
    ra = np.random.uniform(0, 2*np.pi, size=n_data)
    dec = np.pi/2. - np.arccos(2*np.random.uniform(size=n_data)-1.)
    parallax = np.random.uniform(-0.1, 1000., size=n_data)
    pmra,pmdec = np.random.normal(0, 100, size=(2,n_data))

    _M = np.random.uniform(0.1, 0.3, size=(n_data,6,6))
    Cov = 0.5 * np.array([M.dot(M.T) for M in _M])
    _M[:,5] = 0.
    _M[...,5] = 0.
    assert Cov.shape == (n_data,6,6)

    all_data = []
    for r,d,plx,pmr,pmd,C in zip(ra,dec,parallax,pmra,pmdec,Cov):
        row = {
            'ra': np.degrees(r), 'dec': np.degrees(d),
            'parallax': plx, 'pmra': pmr, 'pmdec': pmd
        }
        data = TGASStar(row)
        data._cov = C
        all_data.append(data)

    return all_data

def make_data_increase_uncertainties():
    n_data = 8
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

def make_bad_pairs():
    n_data = 128

    for i in range(n_data):
        ra1,ra2 = np.random.uniform(0, 2*np.pi, size=2)
        dec1,dec2 = np.pi/2. - np.arccos(2*np.random.uniform(size=2)-1.)
        plx1,plx2 = np.exp(np.random.uniform(0,2,size=2))

        true_v = np.random.normal(0, 25, size=3)
        T1 = get_tangent_basis(ra1, dec1)
        v1 = T1.T.dot(true_v)
        pmra1,pmdec1 = v1[:2] * plx1 / 4.740470463496208
        # TODO: ignoring RV for now

        true_v = np.random.normal(0, 25, size=3)
        T2 = get_tangent_basis(ra2, dec2)
        v2 = T2.T.dot(true_v)
        pmra2,pmdec2 = v2[:2] * plx2 / 4.740470463496208
        # TODO: ignoring RV for now

        Cov = np.diag(np.random.uniform(0.1, 0.3, size=6))**2
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

# ----------------------------------------------------------------------------

def test_y():
    all_data = make_random_data()
    for data in all_data:
        d = 1000./data._parallax
        y = get_y(d, data)
        assert y.shape == (4,)
        assert np.allclose(y[0], 0)

    ds = [1000./data._parallax for data in all_data[:2]]
    y = get_y(ds, all_data[:2])
    assert y.shape == (8,)

def test_M():
    all_data = make_random_data()
    for data in all_data:
        M = get_M(data)
        assert M.shape == (4,3)
        assert np.allclose(np.linalg.norm(M, axis=1), [0, 1, 1, 1])

    M = get_M(all_data[:2])
    assert M.shape == (8,3)

    # make sure the projection matrix M is the same for all of these data
    all_data = make_data_increase_uncertainties()
    comp_M = None
    for data in all_data:
        M = get_M(data)
        if comp_M is None:
            comp_M = M
        assert np.allclose(M, comp_M)

def test_Cinv():
    all_data = make_random_data()
    for data in all_data:
        d = 1000./data._parallax
        Cinv = get_Cinv(d, data)
        assert Cinv.shape == (4,4)
        assert np.allclose(Cinv[0], Cinv[0].T)

    ds = [1000./data._parallax for data in all_data[:2]]
    Cinv = get_Cinv(ds, all_data[:2])
    assert Cinv.shape == (8,8)
    assert np.allclose(Cinv[0], Cinv[0].T)

    # as uncertainties increase, determinant of inverse variance matrix should get smaller
    all_data = make_data_increase_uncertainties()
    all_det_Cinv = []
    for data in all_data:
        d = 1000./data._parallax
        Cinv = get_Cinv(d, data)
        sgn,det = np.linalg.slogdet(Cinv[:3,:3])
        all_det_Cinv.append(det)

    all_det_Cinv = np.array(all_det_Cinv)
    assert np.all((all_det_Cinv[1:] - all_det_Cinv[:-1]) < 0)

def test_A_nu_Delta():
    Vinv = np.diag([1/25.**2]*3)

    all_data = make_random_data()
    for data in all_data:
        d = 1000./data._parallax

        M = get_M(data)
        Cinv = get_Cinv(d, data)
        y = get_y(d, data)

        A, nu, Delta = get_A_nu_Delta(d, M, Cinv, y, Vinv)
        assert A.shape == (3,3)
        assert np.isfinite(A).all()

        assert nu.shape == (3,)
        assert np.isfinite(nu).all()

        assert np.isfinite(Delta)

    ds = [1000./data._parallax for data in all_data[:2]]
    M = get_M(all_data[:2])
    Cinv = get_Cinv(ds, all_data[:2])
    y = get_y(ds, all_data[:2])
    A, nu, Delta = get_A_nu_Delta(ds, M, Cinv, y, Vinv)
    assert A.shape == (3,3)
    assert np.isfinite(A).all()
    assert nu.shape == (3,)
    assert np.isfinite(nu).all()
    assert np.isfinite(Delta)

    # as uncertainties increase, nu^T A^-1 nu should *increase*
    all_data = make_data_increase_uncertainties()
    all_nuT_Ainv_nu = []
    all_log_detA = []
    for data in all_data:
        d = 1000./data._parallax
        M = get_M(data)
        Cinv = get_Cinv(d, data)
        y = get_y(d, data)

        A, nu, Delta = get_A_nu_Delta(d, M, Cinv, y, Vinv)
        Ainv = np.linalg.inv(A)
        all_log_detA.append(np.linalg.slogdet(A)[1])
        all_nuT_Ainv_nu.append(nu.dot(Ainv).dot(nu))

    # A should increase as uncertainties increase
    all_log_detA = np.array(all_log_detA)
    assert np.all((all_log_detA[1:] - all_log_detA[:-1]) > 0)

    # nuT_Ainv_nu goes like Sigma^-1, so should decrease
    all_nuT_Ainv_nu = np.array(all_nuT_Ainv_nu)
    assert np.all((all_nuT_Ainv_nu[1:] - all_nuT_Ainv_nu[:-1]) < 0)

def test_ln_H1_marg_likelihood():
    Vinv = np.diag([1/25.**2]*3)
    for pair in make_bad_pairs():
        d1 = 1000/pair[0]._parallax
        d2 = 1000/pair[1]._parallax

        ll = ln_H1_marg_likelihood(d1, d2, pair[0], pair[1], Vinv)
        print(ll)
        return
        assert np.isfinite(ll)

def test_ln_H2_marg_likelihood():
    Vinv = np.diag([1/25.**2]*3)
    for pair in make_bad_pairs():
        d1 = 1000/pair[0]._parallax
        d2 = 1000/pair[1]._parallax

        ll = ln_H2_marg_likelihood(d1, d2, pair[0], pair[1], Vinv)
        assert np.isfinite(ll)

def test_plot_shape_of_Q():
    Vinv = np.diag([1/25.**2]*3)
    for pair in make_good_pairs():
        d1 = 1000/pair[0]._parallax
        print(d1, pair[0]._parallax)
        d_grid = np.linspace(0.1, 8*d1, 256)

        ll = np.zeros_like(d_grid)
        for i,d in enumerate(d_grid):
            ll[i] = ln_H2_marg_likelihood_helper(d, pair[0], Vinv)

        import matplotlib.pyplot as plt
        plt.plot(d_grid, np.exp(ll-ll.max()))
        plt.show()

        return

def test_ln_likelihood_ratio():
    Vinv = np.diag([1/25.**2]*3)
    for pair in make_bad_pairs():
        d1 = 1000/pair[0]._parallax
        d2 = 1000/pair[1]._parallax

        H1 = ln_H1_marg_likelihood(d1, d2, pair[0], pair[1], Vinv)
        H2 = ln_H2_marg_likelihood(d1, d2, pair[0], pair[1], Vinv)
        print(H1 - H2)

        break

    for pair in make_good_pairs():
        d1 = 1000/pair[0]._parallax
        d2 = 1000/pair[1]._parallax

        H1 = ln_H1_marg_likelihood(d1, d2, pair[0], pair[1], Vinv)
        H2 = ln_H2_marg_likelihood(d1, d2, pair[0], pair[1], Vinv)
        print(H1 - H2)

        d1_grid,d2_grid = np.meshgrid(np.linspace(-3, np.log10(16*d1), 128),
                                      np.logspace(-3, np.log10(16*d2), 128))
        d1_grid = d1_grid.ravel()
        d2_grid = d2_grid.ravel()

        HH1 = np.zeros_like(d1_grid)
        HH2 = np.zeros_like(d1_grid)
        for i,(d1,d2) in enumerate(zip(d1_grid, d2_grid)):
            HH1[i] = ln_H1_marg_likelihood(d1, d2, pair[0], pair[1], Vinv)
            HH2[i] = ln_H2_marg_likelihood(d1, d2, pair[0], pair[1], Vinv)

        import matplotlib.pyplot as plt

        plt.figure(figsize=(6,6))
        plt.imshow(HH1.reshape(128,128), cmap='viridis')

        plt.figure(figsize=(6,6))
        plt.imshow(HH2.reshape(128,128), cmap='viridis')

        plt.show()

        break
