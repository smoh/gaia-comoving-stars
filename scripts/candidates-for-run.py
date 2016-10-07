
get_ipython().magic(u'pylab inline')
style.use('notebook.mplstyle')

from astropy import table
from astropy import units as u
from astropy import constants as c
from galpy.util import bovy_coords
from sklearn.neighbors import NearestNeighbors
Table = table.Table

# import sys
# sys.path.append('/Users/semyeong/projects/gaia-wide-binaries')


t = Table.read('/Users/semyeong/data/gaia/tgas_source/stacked_tgas.fits')

snparallax = t['parallax']/t['parallax_error']
# dist = 1./tgas['parallax'].data
# Lutz-Kelker
# beta = 0.
# parallax = t['parallax_error'] * 0.5 * (t['parallax']**2/ t['parallax_error']**2 + \
#                                         sqrt(t['parallax']**2/t['parallax_error']**2 + 4.*beta) )

tgas = t[snparallax>20]
parallax = tgas['parallax'] * (0.5 + 0.5*sqrt(1-(16.*tgas['parallax_error']**2)/tgas['parallax']**2))
dist = 1./parallax.data

gx,gy,gz = bovy_coords.lbd_to_XYZ(
    deg2rad(tgas['l'].data), deg2rad(tgas['b'].data), dist).T



# In[38]:

get_ipython().run_cell_magic(u'time', u'', u"data_gxyz = vstack([gx, gy, gz]).T\nnbrs_gxyz = NearestNeighbors(n_neighbors=10, metric='euclidean').fit(data_gxyz)")


# In[39]:

get_ipython().run_cell_magic(u'time', u'', u'dist_gxyz, indi_gxyz = nbrs_gxyz.kneighbors(data_gxyz, n_neighbors=11)')


# In[40]:


counts_ncandi = []
# for sncut in [10]:
for deltavcut in [1]:
# sncut = 10  # parallax S/N cut
# deltavcut = 5  # km/s

    pm = vstack([tgas['pmra'].data,
                 tgas['pmdec'].data]).T
    dpm_masyr = []
    snp = []
#     bool_snp_atleast1 = []
    for j in range(10):
        ind_jp1 = indi_gxyz[:,j+1]
        pm_jth = vstack([tgas['pmra'][ind_jp1].data,
                         tgas['pmdec'][ind_jp1].data]).T
        dpm_masyr.append(norm(pm-pm_jth, axis=1))
#         snp.append(snparallax[ind_jp1].data)
#         bool_snp_atleast1.append( (snparallax>sncut) & (snparallax[ind_jp1]>sncut) )
    dpm_masyr = array(dpm_masyr).T  # (N, 10) array
#     snp = array(snp).T # (N, 10) array of parallax S/N
#     bool_snp_atleast1 = array(bool_snp_atleast1).T

    dpm_kms = (dpm_masyr.T * dist * 4.74).T

    # apply cut and throw some out
    indi_cand = indi_gxyz[:,1:].copy()
    indi_cand[~((dpm_kms < deltavcut) )] = -1

    Ncand = (indi_cand!=-1).sum(axis=1)
    print 'deltavcut = %2.0f total number of pairs: %8i' % (deltavcut, Ncand.sum())

#     counts, bins = histogram(Ncand[Ncand>0], arange(0.5, 10.6,1),)
#     counts_ncandi.append([sncut, deltavcut, counts])
#         xlabel('number of candidates')
#         ylabel('count')
#         xticks(0.5*(bins[1:] + bins[:-1]))
#         xlim(0,11)
# #         title('sncut = %.0f deltavcut = %.0f' % (sncut, deltavcut))


# In[ ]:

sncut =  5 deltavcut =  5 total number of pairs:   264176
sncut =  5 deltavcut = 10 total number of pairs:   881441
sncut = 10 deltavcut =  5 total number of pairs:   111683
sncut = 10 deltavcut = 10 total number of pairs:   362718
sncut = 20 deltavcut =  5 total number of pairs:    34489
sncut = 20 deltavcut = 10 total number of pairs:   102788
sncut = 40 deltavcut =  5 total number of pairs:     7049
sncut = 40 deltavcut = 10 total number of pairs:    19855


# In[144]:

# indi_cand_ma = ma.MaskedArray(indi_cand, mask=indi_cand==-1)
# indi_cand_ma[:10]


# In[41]:

Ncand.sum()


# In[42]:

sum(Ncand > 0)


# In[43]:

tcand = Table(indi_cand, names=['NN%i' % (i+1) for i in range(10)])


# In[44]:

tcand['Ncand'] = Ncand


# In[45]:

tcand[:10]


# In[46]:

tcand['index0'] = arange(len(t))[snparallax>20]


# In[27]:

len(tcand)


# In[47]:

tcand.write('tgas_pair_indicies_sn20_dv1.fits', overwrite=True)


# In[59]:

sum(tcand['Ncand']>0)


# In[38]:

len(Ncand) - sum(Ncand ==0)


# In[26]:

fig, ax = subplots(5, 2, sharex=True, sharey=True, figsize=(10,20))
fig.subplots_adjust(hspace=0.02, wspace=0.02)
ax = ax.flatten()
for j in range(10):
#     title('%i nearest neighbor' % (jth+1))
    sca(ax[j])
    plot(dist_gxyz[:,j+1]*1e3, dpm_kms[:,j], 'k,', alpha=.3)
    # candidates only
#     plot(dist_gxyz[indi_cand[:,j]!=-1,j+1]*1e3, dpm_kms[indi_cand[:,j]!=-1,j], 'k,')
    xscale('log')
# yscale('log')
    ylim(-3,20)
    xlim(1e-1, 10)


# In[ ]:



