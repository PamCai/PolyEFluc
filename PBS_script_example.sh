#!/bin/bash -l
#PBS -N fluc-jax-iter-unitless
#PBS -l select=1:ngpus=4:system=polaris
##PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=0:60:00
#PBS -q debug-scaling
#PBS -A PolyEFluc
#PBS -l filesystems=home:grand:eagle
#PBS -m bea
#PBS -k doe
#PBS -o polyelectrolyte/stdout/
#PBS -e polyelectrolyte/stderr/

cd ${PBS_O_WORKDIR}

export CUDA_VISIBLE_DEVICES=0,1,2,3

# load modules
module use /soft/modulefiles 
module avail jax
module load jax

# Check GPU availability
nvidia-smi

# Launch script
python polyelectrolyte/PolyEFluc/polyelectrolyte-5-system/theory-rescaled-5comp.py params_5comp1.txt
python polyelectrolyte/PolyEFluc/polyelectrolyte-5-system/theory-rescaled-5comp.py params_5comp2.txt
python polyelectrolyte/PolyEFluc/polyelectrolyte-5-system/theory-rescaled-5comp.py params_5comp3.txt
python polyelectrolyte/PolyEFluc/polyelectrolyte-5-system/theory-rescaled-5comp.py params_5comp4.txt
python polyelectrolyte/PolyEFluc/polyelectrolyte-5-system/theory-rescaled-5comp.py params_5comp5.txt
python polyelectrolyte/PolyEFluc/polyelectrolyte-5-system/theory-rescaled-5comp.py params_5comp6.txt
python polyelectrolyte/PolyEFluc/polyelectrolyte-5-system/theory-rescaled-5comp.py params_5comp7.txt
python polyelectrolyte/PolyEFluc/polyelectrolyte-5-system/theory-rescaled-5comp.py params_5comp8.txt
