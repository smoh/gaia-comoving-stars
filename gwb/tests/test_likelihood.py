from __future__ import division, print_function

# Third-party
import astropy.units as u
import astropy.coordinates as coord
import numpy as np

# Project
from ..likelihood import (get_u_vec, get_tangent_basis,
                          ln_H1_marg_likelihood, ln_H2_marg_likelihood)

n_test = 8

def test_u_vec():
    ra = np.random.uniform(0, 2*np.pi, size=n_test)
    dec = np.pi/2. - np.arccos(2*np.random.uniform(size=n_test)-1.)

    assert get_u_vec(ra[0], dec[0]).shape == (3,)
    assert get_u_vec(ra, dec).shape == (n_test,3)

    for lon,lat in zip(ra,dec):
        usph = coord.UnitSphericalRepresentation(lon=lon*u.rad, lat=lat*u.rad)
        astropy_xyz = usph.represent_as(coord.CartesianRepresentation).xyz.value.T
        assert np.allclose(astropy_xyz, get_u_vec(lon, lat))


def test_tangent_basis():
    ra = np.random.uniform(0, 2*np.pi, size=n_test)
    dec = np.pi/2. - np.arccos(2*np.random.uniform(size=n_test)-1.)

    for a,d in zip(ra, dec):
        A = get_tangent_basis(a, d)
        assert A.shape == (3,3)

        assert np.allclose(np.linalg.norm(A, axis=1), 1.)
        assert np.allclose(np.dot(A[0], A[1]), 0.)
        assert np.allclose(np.dot(A[0], A[2]), 0.)
        assert np.allclose(np.dot(A[1], A[2]), 0.)

        assert np.allclose(np.cross(A[0], A[1]), A[2])
        assert np.allclose(np.cross(A[0], A[2]), -A[1])
        assert np.allclose(np.cross(A[1], A[2]), A[0])

def make_pair_same_v(d1=81.314, sep_pc=0.1):
    # same velocity vector
    true_v = np.random.normal(0, 30., size=3)

    ra1 = np.random.uniform(0, 2*np.pi)
    dec1 = np.pi/2. - np.arccos(2*np.random.uniform() - 1.)
    v1 = get_tangent_basis(ra1, dec1).dot(true_v)

    ra2 = ra1 + np.random.choice([1.,-1.])*sep_pc/np.sqrt(2.)/d1
    dec_choice = np.random.choice([1.,-1.])
    dec2 = dec1 + dec_choice*sep_pc/np.sqrt(2.)/d1
    if dec2 > np.pi/2 or dec2 < -np.pi/2.:
        dec2 += -2*dec_choice*sep_pc/np.sqrt(2.)/d1

    d2 = d1 + np.random.choice([1.,-1.])*sep_pc/np.sqrt(2.)
    v2 = get_tangent_basis(ra2, dec2).dot(true_v)

    # picked at random from TGAS data
    Cov1 = np.array([[0.066019, 0.0581179, -0.06287875],
                     [0.0581179, 2.76016904, -1.31836247],
                     [-0.06287875, -1.31836247, 0.83536227]])
    Cov2 = np.array([[0.07143981, 0.31518988, -0.11283286],
                     [0.31518988, 6.78049538, -2.21299533],
                     [-0.11283286, -2.21299533, 0.9342514]])
    # Cov1 = Cov2 = np.diag(np.zeros(3) + 1E-8)

    plx1 = (d1*u.pc).to(u.mas,u.parallax()).value
    plx2 = (d2*u.pc).to(u.mas,u.parallax()).value

    pm1 = (v1[:2] / d1 * u.km/u.s/u.pc).to(u.mas/u.yr, u.dimensionless_angles()).value
    pm2 = (v2[:2] / d2 * u.km/u.s/u.pc).to(u.mas/u.yr, u.dimensionless_angles()).value

    true_x1 = np.concatenate(([plx1],pm1))
    true_x2 = np.concatenate(([plx2],pm2))

    x1 = np.zeros(4) # ignore vr
    x1[:3] = np.random.multivariate_normal(true_x1, Cov1)

    x2 = np.zeros(4) # ignore vr
    x2[:3] = np.random.multivariate_normal(true_x2, Cov2)

    Cinv1 = np.zeros((4,4))
    Cinv1[:3,:3] = np.linalg.inv(Cov1)

    Cinv2 = np.zeros((4,4))
    Cinv2[:3,:3] = np.linalg.inv(Cov2)

    return (ra1,dec1,x1,Cinv1), (ra2,dec2,x2,Cinv2), true_x1, true_x2, true_v

def test_fake_data_same():

    for d1, sep in zip(np.random.exponential(scale=200, size=n_test), np.logspace(-3, 0.5, n_test)):
        star1, star2, true1, true2, true_v = make_pair_same_v(d1=d1, sep_pc=sep)

        lnH1 = ln_H1_marg_likelihood(true_v, 1000./true1[0], 1000./true2[0], star1, star2)
        lnH2 = ln_H2_marg_likelihood(true_v, np.random.normal(0,30,3),
                                     1000./true1[0], 1000./true2[0],
                                     star1, star2)

        assert lnH1 > lnH2

    print("-"*42)

def make_pair_diff_v(d1=81.314, sep_pc=0.1):
    # same velocity vector
    true_v1 = np.random.normal(0, 30., size=3)
    true_v2 = np.random.normal(0, 30., size=3)

    ra1 = np.random.uniform(0, 2*np.pi)
    dec1 = np.pi/2. - np.arccos(2*np.random.uniform() - 1.)
    v1 = get_tangent_basis(ra1, dec1).dot(true_v1)

    ra2 = ra1 + np.random.choice([1.,-1.])*sep_pc/np.sqrt(2.)/d1
    dec_choice = np.random.choice([1.,-1.])
    dec2 = dec1 + dec_choice*sep_pc/np.sqrt(2.)/d1
    if dec2 > np.pi/2 or dec2 < -np.pi/2.:
        dec2 += -2*dec_choice*sep_pc/np.sqrt(2.)/d1

    d2 = d1 + np.random.choice([1.,-1.])*sep_pc/np.sqrt(2.)
    v2 = get_tangent_basis(ra2, dec2).dot(true_v2)

    # picked at random from TGAS data
    Cov1 = np.array([[0.066019, 0.0581179, -0.06287875],
                     [0.0581179, 2.76016904, -1.31836247],
                     [-0.06287875, -1.31836247, 0.83536227]])
    Cov2 = np.array([[0.07143981, 0.31518988, -0.11283286],
                     [0.31518988, 6.78049538, -2.21299533],
                     [-0.11283286, -2.21299533, 0.9342514]])
    # Cov1 = Cov2 = np.diag(np.zeros(3) + 1E-8)

    plx1 = (d1*u.pc).to(u.mas,u.parallax()).value
    plx2 = (d2*u.pc).to(u.mas,u.parallax()).value

    pm1 = (v1[:2] / d1 * u.km/u.s/u.pc).to(u.mas/u.yr, u.dimensionless_angles()).value
    pm2 = (v2[:2] / d2 * u.km/u.s/u.pc).to(u.mas/u.yr, u.dimensionless_angles()).value

    true_x1 = np.concatenate(([plx1],pm1))
    true_x2 = np.concatenate(([plx2],pm2))

    x1 = np.zeros(4) # ignore vr
    x1[:3] = np.random.multivariate_normal(true_x1, Cov1)

    x2 = np.zeros(4) # ignore vr
    x2[:3] = np.random.multivariate_normal(true_x2, Cov2)

    Cinv1 = np.zeros((4,4))
    Cinv1[:3,:3] = np.linalg.inv(Cov1)

    Cinv2 = np.zeros((4,4))
    Cinv2[:3,:3] = np.linalg.inv(Cov2)

    return (ra1,dec1,x1,Cinv1), (ra2,dec2,x2,Cinv2), true_x1, true_x2, true_v1, true_v2

def test_fake_data_diff():

    for d1, sep in zip(np.random.exponential(scale=200, size=n_test), np.logspace(-3, 0.5, n_test)):
        star1, star2, true1, true2, true_v1, true_v2 = make_pair_diff_v(d1=d1, sep_pc=sep)

        lnH1 = ln_H1_marg_likelihood(true_v1, 1000./true1[0], 1000./true2[0], star1, star2)
        lnH2 = ln_H2_marg_likelihood(true_v1, true_v2,
                                     1000./true1[0], 1000./true2[0],
                                     star1, star2)
        assert lnH1 < lnH2
