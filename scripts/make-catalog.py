#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: make-catalog.py
Author: Semyeong Oh
Github: https://github.com/smoh
Description: Make the final catalog of co-moving pairs
"""

import numpy as np
import networkx as nx
import h5py
from astropy.io import fits
from astropy import table
from astropy import coordinates as coords
import astropy.units as u

import gwb

tgas = gwb.TGASData('data/stacked_tgas.fits')
pairidx = fits.getdata('output/23560/snr8_r10_dv10.fits')
with h5py.File("output/23560/snr8_r10_dv10_vscatter0-lratio.h5") as f:
    lnH1 = f['lnH1'].value
    lnH2 = f['lnH2'].value
    llr = lnH1 - lnH2
print('number of pairs', pairidx.size)

parallax_snr = tgas.parallax_snr
vtan = tgas.get_vtan().value
c = tgas.get_coord()
d = tgas.get_distance().value

star1, star2 = pairidx['star1'], pairidx['star2']
min_snr = np.min(np.vstack((parallax_snr[star1], parallax_snr[star2])), axis=0)
dvtan = np.linalg.norm(vtan[star1]-vtan[star2], axis=1)
vtanmean = (vtan[star1] + vtan[star2])*0.5
sep = c[star1].separation_3d(c[star2]).value
sep_sky = c[star1].separation(c[star2])

c1 = c[star1]
c2 = c[star2]
ra1, dec1 = c1.ra.value, c1.dec.value
ra2, dec2 = c2.ra.value, c2.dec.value
l1, b1 = c1.transform_to(coords.Galactic).l.value, c1.transform_to(coords.Galactic).b.value
l2, b2 = c2.transform_to(coords.Galactic).l.value, c2.transform_to(coords.Galactic).b.value
d1 = d[star1]
d2 = d[star2]
dmean = (d1+d2)*0.5

cond_lr_cut = llr>6
print('number of pairs ln(L1/L2)>6 = %i' % (cond_lr_cut.sum()))
print('number of pairs ln(L1/L2)>6 + sep<1pc = %i' % (sum((sep<1)&cond_lr_cut)))
cmpairs = pairidx[cond_lr_cut]

# sorted list of subgraphs from largest to smallest
graph = nx.from_edgelist(
    [(i,j) for i,j in zip(cmpairs['star1'],cmpairs['star2'])])
Gc = np.array(sorted(nx.connected_component_subgraphs(graph), key=len, reverse=True))
sizes = np.array([len(g) for g in Gc])
nsize = np.array([len(nx.node_connected_component(graph, s1)) for s1 in cmpairs['star1']])

# get group index for each star1
gid = []
def get_nid(node):
    out = []
    for i, g in enumerate(Gc):
        if node in g:
            out.append(i)
    if len(out)>1:
        raise ValueError('This cannot happen.')
    return out[0]
gid = [get_nid(i) for i in cmpairs['star1']]

tt = table.Table()
# tt['star1'] = cmpairs['star1']
# tt['star2'] = cmpairs['star2']
tt['star1 source id'] = tgas.source_id[cmpairs['star1']]
tt['star1 source id'].unit = '--'
tt['star2 source id'] = tgas.source_id[cmpairs['star2']]
tt['star2 source id'].unit = '--'
tt['Sep'] = cmpairs['sep']
tt['Sep'].format='%9.1f'
tt['Sep'].unit = 'pc'
tt['lnL1/L2'] = llr[cond_lr_cut]
tt['lnL1/L2'].format='%9.1f'
tt['lnL1/L2'].unit='--'
tt['N_CC'] = nsize
tt['ID_CC'] = gid
# tt['RAVE_OBS_ID1'] = rave['RAVE_OBS_ID'][cmpairs['star1']]
# tt['RAVE_OBS_ID2'] = rave['RAVE_OBS_ID'][cmpairs['star2']]


# Add information of closest MWSC
mwsc = table.Table.read('data/J_A+A_585_A101/catalog.dat', readme='data/J_A+A_585_A101/ReadMe',
                  format='ascii.cds')
print('total number of mwsc', len(mwsc))
print('number of mwsc d<600 pc', (mwsc['d']<600).sum())
ccm1 = c[cmpairs['star1']]
c_mwsc = coords.SkyCoord(mwsc['GLON'], mwsc['GLAT'], mwsc['d'], frame=coords.Galactic)
idx_mwsc, sep2d_mwsc, dist3d_mwsc = ccm1.match_to_catalog_3d(c_mwsc)
tt['ID_MWSC_closest'] = idx_mwsc
tt['d_MWSC_closest'] = dist3d_mwsc
tt['d_MWSC_closest'].format='%9.1f'

# Match to de Zeeuw by HIP id
obass = table.Table.read('data/J_AJ_117_354/tablec1.dat', readme='data/J_AJ_117_354/ReadMe',
                    format='ascii.cds')
print('number of OB association stars', len(obass))

# query simbad on HIP id's to get coordinates
from astroquery.simbad import Simbad
customSimbad = Simbad()
customSimbad.add_votable_fields('sptype', 'parallax')
result = customSimbad.query_objects(['HIP %i' % hip for hip in obass['HIP']])
print( np.unique([s.decode("utf-8")[0] if len(s)>0 else '?' for s in result['SP_TYPE']]) )

def get_distance(parallax, parallax_error):
    """
    Return the distance [kpc] point estimate with the Lutz-Kelker correction
    
    parallax : float, in mas
    parallax_error : float, in mas
    """
    snr = parallax / parallax_error
    pnew = parallax * (0.5 + 0.5*np.sqrt(1 - 16./snr**2))
    # if snr<4, the value will be maksed
    return 1./pnew

obass_dist = get_distance(result['PLX_VALUE'], result['PLX_ERROR'])
obass_c = coords.SkyCoord(result['RA'], result['DEC'], unit=(u.hourangle, u.deg),
                          distance=obass_dist*u.kpc)
obass_cg = obass_c.transform_to(coords.Galactic)

t1 = table.Table([tgas.hip[cmpairs['star1']]], names=['HIP'])
t1_obass = table.join(obass['OBAss','HIP'], t1, keys='HIP', join_type='right')
t2 = table.Table([tgas.hip[cmpairs['star2']]], names=['HIP'])
t2_obass = table.join(obass['OBAss','HIP'], t1, keys='HIP', join_type='right')

tt['star1 OB Assc.'] = t1_obass['OBAss']
tt['star2 OB Assc.'] = t2_obass['OBAss']

tt.write('catalog.csv', format='ascii.no_header', delimiter=',')
