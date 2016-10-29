import os

DATADIR = os.getenv('GWBDATA', os.path.expanduser('~/.gwb'))
STARMODELDIR = os.path.join(DATADIR, 'starmodels')
TGASFILE = os.path.join(DATADIR, 'TgasSource.h5')
AVFILE = os.path.join(DATADIR, 'AVs.txt')
XMATCHFILE = os.path.join(DATADIR, 'isochrone_xmatch.vot')
