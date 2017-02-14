#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: make-catalog.py
Author: Semyeong Oh
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

if 'data_loaded' not in dir():
    data_loaded = True
    tgas = gwb.TGASData('data/stacked_tgas.fits')
    pairidx = fits.getdata('output/23560/snr8_r10_dv10.fits')
    with h5py.File("output/23560/snr8_r10_dv10_vscatter0-lratio.h5") as f:
        lnH1 = f['lnH1'].value
        lnH2 = f['lnH2'].value
        llr = lnH1 - lnH2
    print('number of pairs', pairidx.size)

    parallax_snr = tgas.parallax_snr
    vtan = tgas.get_vtan().value
    tgas_c = tgas.get_coord()
    tgas_d = tgas.get_distance().to(u.pc).value

    cond_lr_cut = llr>6
    print('number of pairs ln(L1/L2)>6 = %i' % (cond_lr_cut.sum()))
    cmpairs = pairidx[cond_lr_cut]

    # make graph with edge attribute for ln(L1/L2)
    graph = nx.Graph()
    graph.add_edges_from(
        [(i,j, {'lnL1/L2':ll}) for (i,j,dvtan,s), ll in zip(cmpairs, llr[llr>6])])

    # assign id to connected components in descending order of size
    # and set node attribute
    components = sorted(nx.connected_component_subgraphs(graph), key=len, reverse=True)
    comp_dict = {idx: comp.nodes() for idx, comp in enumerate(components)}
    attr = {n: comp_id for comp_id, nodes in comp_dict.items() for n in nodes}
    nx.set_node_attributes(graph, "group_id", attr)

    # 2mass
    tmass = fits.getdata('data/tgas_tmassj.fits', ext=1)
    jhk = table.Table.read('notebooks/result.vot', format='votable')

    # RAVE
    rave = table.Table.read('data/tgas_rave_RV.fits')
    tgas_has_rave = ~rave['row_id'].mask
    tgas_rave_obs_id = rave['RAVE_OBS_ID']
    tgas_rave_hrv = rave['HRV']
    tgas_rave_ehrv = rave['eHRV']

tStar = table.Table()
tStar['tgas_row'] = graph.nodes()
tStar['tgas_source_id'] = tgas.source_id[tStar['tgas_row']]
tStar['tgas_ra'] = tgas.ra.value[tStar['tgas_row']]
tStar['tgas_dec'] = tgas.dec.value[tStar['tgas_row']]
tStar['tgas_parallax'] = tgas.parallax.value[tStar['tgas_row']]
tStar['tgas_distance'] = tgas_d[tStar['tgas_row']]
tStar['tgas_gmag'] = tgas._data['phot_g_mean_mag'][tStar['tgas_row']]
tStar['tmass_jmag'] = tmass['j_m'][tStar['tgas_row']]
tStar['rave_obs_id'] = tgas_rave_obs_id[tStar['tgas_row']]
tStar['rave_hrv'] = tgas_rave_hrv[tStar['tgas_row']]
tStar['rave_ehrv'] = tgas_rave_ehrv[tStar['tgas_row']]
tStar['group_id'] =[graph.node.get(n)['group_id'] for n in tStar['tgas_row']]
tStar['group_size'] = [len(nx.node_connected_component(graph, node)) for node in tStar['tgas_row']]

star_c = tgas_c[tStar['tgas_row']]

tPair = table.Table()
edge_tgas_row = np.array(graph.edges())
edge_star_row = array([list(map(lambda node: where(tStar['tgas_row'] == node)[0][0], [star1, star2])) for star1, star2 in edge_tgas_row])
tPair['tgas_row1'] = edge_tgas_row[:,0]
tPair['tgas_row2'] = edge_tgas_row[:,1]
tPair['star_row1'] = edge_star_row[:,0]
tPair['star_row2'] = edge_star_row[:,1]
tPair['angsep'] = star_c[tPair['star_row1']].separation(star_c[tPair['star_row2']]).to(u.arcmin).value
tPair['separation'] = star_c[tPair['star_row1']].separation_3d(star_c[tPair['star_row2']]).to(u.pc).value
tPair['lnL1/L2'] = np.array([graph.get_edge_data(i,j)['lnL1/L2'] for i,j in edge_tgas_row])

tPair['group_id'] = tStar['group_id'][tPair['star_row1']]
tPair['group_size'] = tStar['group_size'][tPair['star_row1']]

tGroup = table.Table()
tGroup['id'] = np.array([i for i in comp_dict.keys()])
tGroup['size'] = np.array([len(comp_dict[i]) for i in tGroup['id']])
tGroup['mean_ra'] = np.array([ np.mean(tStar['tgas_ra'][tStar['group_id']==i]) for i in tGroup['id']])
tGroup['mean_dec'] = np.array([ np.mean(tStar['tgas_dec'][tStar['group_id']==i]) for i in tGroup['id']])
tGroup['mean_dist'] = np.array([ np.mean(tStar['tgas_distance'][tStar['group_id']==i]) for i in tGroup['id']])

# mwsc = table.Table.read('data/J_A+A_585_A101/catalog.dat', readme='data/J_A+A_585_A101/ReadMe',
#                  format='ascii.cds')
# print('total number of mwsc', len(mwsc))
# print('number of mwsc d<600 pc', (mwsc['d']<600).sum())

# group_c = coords.SkyCoord(group_mean_ra*u.deg, group_mean_dec*u.deg, group_mean_dist*u.pc)
# mwsc_c = coords.SkyCoord(mwsc['GLON'], mwsc['GLAT'], distance=mwsc['d'].to(u.pc), frame=coords.Galactic,) 

# group_mwsc_match = group_c.match_to_catalog_3d(mwsc_c,)

tStar.write('table_star.csv', format='ascii.csv')
tPair.write('table_pair.csv', format='ascii.csv')
tGroup.write('table_group.csv', format='ascii.csv')