#!/bin/bash
#SBATCH -J likelihood-ratio           # job name
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

export SNR=4
export NEIGH=128
export DV=10

git log -1

OUTDIR="output/$SLURM_JOBID"
echo `date`
echo "SNR=$SNR NEIGHBOR=$NEIGH DV=$DV"
echo "OUTDIR=$OUTDIR"
mkdir $OUTDIR

python scripts/generate-pair-sample.py --tgas-file=data/stacked_tgas.fits \
--snr-cut=$SNR \
--output-path $OUTDIR \
neighbors \
-n $NEIGH \
--dv=$DV
echo "pair indicies created"

srun python scripts/compute-likelihood-ratio.py --tgas-file=data/stacked_tgas.fits -v --mpi \
--pairs-file=$OUTDIR/snr"$SNR"_n"$NEIGH"_dv"$DV".fits \
--vscatter=0
