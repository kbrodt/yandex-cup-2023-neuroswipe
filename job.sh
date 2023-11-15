#!/bin/bash
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64000M
#SBATCH --time=1-00:00          # DD-HH:MM:SS


SOURCEDIR=~/neuroswipe
data_dir=$SLURM_TMPDIR/data

module load python/3.11.5
#cuda cudnn
source ~/env/bin/activate

#virtualenv --no-download $SLURM_TMPDIR/env
#source $SLURM_TMPDIR/env/bin/activate
#pip install --no-index --upgrade pip setuptools
#pip install --no-index -r $SOURCEDIR/requirements.txt

sh $SOURCEDIR/run.sh
