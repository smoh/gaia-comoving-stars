
[![Build Status](https://travis-ci.com/smoh/gaia-comoving-stars.svg?token=snqHW5KtLdPNJV6qzcFr&branch=master)](https://travis-ci.com/smoh/gaia-comoving-stars)

# Co-moving stars in Gaia DR1

This repository contains code used for finding co-moving pairs of stars using the TGAS catalog. The paper is posted on [arxiv](https://arxiv.org/abs/1612.02440).

An interactive web visualization is available [here](www.smoh.space/vis/gaia-comoving-stars/).

## Link to the catalog in CSV format:

- [star table](https://raw.githubusercontent.com/smoh/gaia-comoving-stars/master/paper/t1-1-star.txt)
- [pair table](https://raw.githubusercontent.com/smoh/gaia-comoving-stars/master/paper/t1-2-pair.txt)
- [group table](https://raw.githubusercontent.com/smoh/gaia-comoving-stars/master/paper/t1-3-group.txt)

## Data

You can download the stacked Tycho-Gaia data as a FITS file from
[this URL](http://s3.adrian.pw/stacked_tgas.fits) or by doing:

```bash
wget http://s3.adrian.pw/stacked_tgas.fits
```

## What's in the code?

- README.md: this file
- environment.yml: [conda environment](https://conda.io/docs/using/envs.html#clone-an-environment) specification.
  `conda env create` will read this file and install necessary packages in an environment named `gwb`.
  Do `source activate gwb` to activate the environment.
- gwb/ : main source code
- mpi/ : contains example job scripts for computing clusters
- notebooks/ : contains notebooks used to analyze results
- paper/ : paper TeX files
- scripts/ : contains scripts used to generate pair sample, and calculate likelihood ratios
- setup.py: python install script. In order to install the code in development mode, do `python setup.py develop`.
  This way, changes you make in the code base is reflected immediately without the need to install again.


## License

Copyright 2016 the Authors. Licensed under the terms of [the MIT
License](https://github.com/smoh/gaia-wide-binaries/blob/master/LICENSE).
