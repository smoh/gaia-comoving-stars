#!/bin/bash
#SBATCH -J vshuffle                # job name
#SBATCH -o logs/ll.o%j             # output file name (%j expands to jobID)
#SBATCH -e logs/ll.e%j             # error file name (%j expands to jobID)
#SBATCH -n 256                  # total number of mpi tasks requested
#SBATCH -t 02:30:00             # run time (hh:mm:ss) - 1.5 hours
#SBATCH --mail-user=semyeong@princeton.edu
#SBATCH --mail-type=begin       # email me when the job starts
#SBATCH --mail-type=end         # email me when the job finishes

cd $HOME/projects/gaia-wide-binaries

module load openmpi/gcc/1.10.2/64

source activate gwb

git log -1
echo `date`

OUTDIR="output/$SLURM_JOBID"
echo "OUTDIR=$OUTDIR"
mkdir $OUTDIR

SNR=8
RADIUS_CUT=10
RADIUS_CUT_VSHUF=10
DVTAN_CUT=10
OUTNAME=OUTDIR/"snr${SNR}_r${RADIUS_CUT}_rv${RADIUS_CUT_VSHUF}_dvtan${DVTAN_CUT}.h5"
echo "output: ${OUTNAME}"

srun python scripts/test_vshuffle.py \
  --tgas-file data/stacked_tgas.fits \
  --outname $OUTNAME \
  --snr-cut $SNR \
  --radius-cut $RADIUS_CUT \
  --radius-cut-vshuf $RADIUS_CUT_VSHUF \
  --dvtan-cut $DVTAN_CUT \
  --mpi
