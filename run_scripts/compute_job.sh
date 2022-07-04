#!/bin/bash
#SBATCH --job-name=ALFsharedtrial
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jfm343@cornell.edu
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=124G
#SBATCH -t 48:00:00
#SBATCH --account=crl171
#SBATCH --output="job.%j.%N.out"
#SBATCH --error="job.%j.%N.err"
#SBATCH --export=ALL


module purge
module load cpu

#Load module file(s) into the shell environment
module load gcc openblas openmpi hdf5
module load slurm
module load matlab
module load anaconda3/2020.11


#running
#lattice filling angle kappa HFmode polarization
srun python3 -u Bubble_ep.py 30 0 ${param_val} 0.7 0 T

##moving data out
date_fin="`date "+%Y-%m-%d-%H-%M-%S"`"
dirfin="${PRDIR}/${SLURM_JOB_NAME}_${date_fin}"
mkdir "${dirfin}"
mv * "${dirfin}"
