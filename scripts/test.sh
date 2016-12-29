
# takes about 35s to run
# this will produce 58 pairs
python scripts/generate-pair-sample.py -o --tgas-file data//stacked_tgas.fits --snr-cut 80 \
  --outname snr80_n1_dv1.fits neighbors -n 1 --dv 1

echo "pairs file created"

# takes about 2min
#python scripts/compute-likelihood-ratio.py --tgas-file data/stacked_tgas.fits \
#  --pairs-file snr80_n1_dv1.fits \
#  --output-path .
