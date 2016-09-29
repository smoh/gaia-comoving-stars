from __future__ import division, print_function

# Third-party
import astropy.units as u
import astropy.coordinates as coord
import numpy as np

# Project
from ..gaiatools import TGASData
from ..likelihood import get_y, get_M, get_Cinv, get_A_mu_Delta

# make fake data
n_data = 128
ra = np.random.uniform(0, 2*np.pi, size=n_data)
dec = np.pi/2. - np.arccos(2*np.random.uniform(size=n_data)-1.)
parallax = np.random.uniform(-0.1, 1000., size=n_data)
pmra,pmdec = np.random.normal(-0, 100, size=(2,n_data))

np.random.uniform([0.1,])
