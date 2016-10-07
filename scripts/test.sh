
# takes about 35s to run
# this will produce 58 pairs
python scripts/generate-pair-sample.py -o --tgas-file ~/data/gaia/tgas_source/stacked_tgas.fits --snr-cut 80 --output-path . neighbors -n 1 --dv 1

echo "pairs file created"

# takes about 2min
python scripts/compute-likelihood-ratio.py --tgas-file ~/data/gaia/tgas_source/stacked_tgas.fits \
  --pairs-file snr80_n1_dv1.fits \
  --output-path .
