
[![Build Status](https://travis-ci.com/smoh/gaia-comoving-stars.svg?token=snqHW5KtLdPNJV6qzcFr&branch=master)](https://travis-ci.com/smoh/gaia-comoving-stars)

# Co-moving stars in Gaia DR1

This repository contains code used for finding co-moving pairs of stars using the TGAS catalog. The paper is posted on [arxiv](https://arxiv.org/abs/1612.02440).

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

## License

Copyright 2016 the Authors. Licensed under the terms of [the MIT
License](https://github.com/smoh/gaia-wide-binaries/blob/master/LICENSE).
