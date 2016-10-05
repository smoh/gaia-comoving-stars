#!/bin/bash
#SBATCH -J likelihood-ratio           # job name
#SBATCH -o h1h2.o%j             # output file name (%j expands to jobID)
#SBATCH -e h1h2.e%j             # error file name (%j expands to jobID)
#SBATCH -n 256                  # total number of mpi tasks requested
#SBATCH -t 01:00:00             # run time (hh:mm:ss) - 1.5 hours
#SBATCH --mail-user=adrn@princeton.edu
#SBATCH --mail-type=begin       # email me when the job starts
#SBATCH --mail-type=end         # email me when the job finishes

cd /tigress/adrianp/projects/gaia-wide-binaries/scripts

module load openmpi/gcc/1.10.2/64

source activate gwb

export SNR=8
export NEIGH=64
export DV=8

python generate-pair-sample.py --tgas-file=../data/stacked_tgas.fits \
--snr-cut=$SNR \
--neighbors=$NEIGH \
--deltav-cut=$DV

srun python compute-likelihood-ratio.py --tgas-file=../data/stacked_tgas.fits -v --mpi \
--snr-cut=$SNR \
--pairs-file=../data/snr"$SNR"_n"$NEIGH"_dv"$DV".fits \
--vscatter=0
