import pandas as pd
import numpy as np

from .phot import get_photometry

try:
    from isochrones.starmodel import ResolvedBinaryStarModel
    from isochrones.mist import MIST_Isochrone
    from isochrones.extinction import get_AV_infinity
except ImportError:
    class ResolvedBinaryStarModel(object):
        pass
        
    logging.warning('isochrones not installed.')

# Globals
tgas = None
mist = None

def get_source_id(idx):
    """Retursn source_id from tgas stacked index
    """
    global tgas
    if tgas is None:
        from .cfg import TGASFILE
        tgas = pd.read_hdf(TGASFILE, 'df')

    return tgas.iloc[idx].source_id

def get_radec(idx):
    global tgas
    if tgas is None:
        from .cfg import TGASFILE
        tgas = pd.read_hdf(TGASFILE, 'df')

    return tgas.iloc[idx].ra, tgas.iloc[idx].dec

class TGASWideBinaryStarModel(ResolvedBinaryStarModel):
    def __init__(self, ic, idx1, idx2):
        self.idx1 = idx1
        self.idx2 = idx2
                
        i1 = min(idx1, idx2)
        i2 = max(idx1, idx2)
        name = '{}-{}'.format(i1,i2)
        
        # Get x-matched photometry
        star1 = get_photometry(get_source_id(idx1))
        star2 = get_photometry(get_source_id(idx2))
        
        ra1, dec1 = get_radec(idx1)
        ra2, dec2 = get_radec(idx2)

        AV1 = get_AV_infinity(ra1, dec1)
        AV2 = get_AV_infinity(ra2, dec2)

        super(TGASWideBinaryStarModel, self).__init__(ic, star1, star2, 
                                                    name=name, maxAV=max(AV1, AV2))
        