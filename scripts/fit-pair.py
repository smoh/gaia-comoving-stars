#!/usr/bin/env python
from __future__ import print_function, division
import sys, os
import time

# To make figures on compute servers
import matplotlib
matplotlib.use('AGG')

from gwb.starmodels import TGASWideBinaryStarModel, STARMODELDIR

star1 = int(sys.argv[1])
star2 = int(sys.argv[2])

model_dir = os.path.join(STARMODELDIR, 'models')
fig_physical_dir = os.path.join(STARMODELDIR, 'figures', 'physical')
fig_observed_dir = os.path.join(STARMODELDIR, 'figures', 'observed')
for d in [model_dir, fig_physical_dir, fig_observed_dir]:
    if not os.path.exists(d):
        os.makedirs(d)

mod = TGASWideBinaryStarModel(star1, star2)

try:
    mod.obs.print_ascii()
except:
    pass

sys.stdout.write('Fitting binary star model for {}...'.format(mod.name))
start = time.time()
mod.fit()
end = time.time()
sys.stdout.write('Fit took {:.2f} min.\n'.format((end-start)/60.))

sys.stdout.write('Writing corner plots...')
fig1 = mod.corner_physical()
fig1.savefig(os.path.join(fig_physical_dir, '{}-physical.png'.format(mod.name)))
fig2 = mod.corner_observed()
fig2.savefig(os.path.join(fig_observed_dir, '{}-observed.png'.format(mod.name)))
h5file = os.path.join(model_dir, '{}.h5'.format(mod.name))
mod.save_hdf(h5file)
sys.stdout.write('Done.  Starmodel saved to {}.\n'.format(h5file))
