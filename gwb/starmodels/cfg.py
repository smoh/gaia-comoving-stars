import os

DATADIR = os.getenv('GWBDATA', os.path.expanduser('~/.gwb'))
STARMODELDIR = os.path.join(DATADIR, 'starmodels')
