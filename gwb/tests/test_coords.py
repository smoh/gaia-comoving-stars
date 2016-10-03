from __future__ import division, print_function

# Third-party
import astropy.units as u
import astropy.coordinates as coord
import numpy as np

# Project
from ..coords import get_u_vec, get_tangent_basis

n_test = 128

def test_u_vec():
    ra = np.random.uniform(0, 2*np.pi, size=n_test)
    dec = np.pi/2. - np.arccos(2*np.random.uniform(size=n_test)-1.)

    assert get_u_vec(ra[0], dec[0]).shape == (3,)
    assert get_u_vec(ra, dec).shape == (3,n_test)

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

    A = get_tangent_basis(ra[:4], dec[:4])
    assert A.shape == (4,3,3)

    assert np.allclose(np.linalg.norm(A, axis=1), 1.)
    assert np.allclose(np.einsum('ij,ij->', A[:,0], A[:,1]), 0.)
    assert np.allclose(np.einsum('ij,ij->', A[:,0], A[:,2]), 0.)
    assert np.allclose(np.einsum('ij,ij->', A[:,1], A[:,2]), 0.)

    assert np.allclose(np.cross(A[:,0], A[:,1]), A[:,2])
    assert np.allclose(np.cross(A[:,0], A[:,2]), -A[:,1])
    assert np.allclose(np.cross(A[:,1], A[:,2]), A[:,0])

def test_projections():
    # Should be all radial
    all_radial = [
        (0,0,[100.,0,0]),
        (0,90,[0,0,100]),
        (90,0,[0,100,0]),
        (-90,0,[0,-100,0])
    ]
    for ra,dec,v in all_radial[-1:]:
        T = get_tangent_basis(np.radians(ra), np.radians(dec))
        assert np.allclose(T.dot(v), [0,0,100])

