#!/bin/bash
# sbatch directives
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=13
#SBATCH --account=morpheus-gpu-rw
#SBATCH --partition=morpheus-gpu-rw
#SBATCH --gres=gpu
##SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=logan.gibb@utah.edu
#SBATCH -o calcination-%j.out-%N

WORKDIR=/uufs/chpc.utah.edu/common/HIPAA/proj_morpheus/logan/calcination/
SCRDIR=/scratch/local/$USER/$SLURM_JOBID/working

echo making directory
mkdir -p $SCRDIR

echo purging module and loading new
module purge
module use $HOME/MyModules
module load miniconda3/latest
conda activate base

echo coping data at $(date)
cp -r $WORKDIR/* $SCRDIR
cd $SCRDIR/training

echo starting main
#pip install --user -r requirements.txt
python main.py -lr1 1e-4 -fe 100 -lds 1 -ldr 1 -k 5 -ds "Calcination" -bse true -mag 10000 50000 100000 -adam true -p true -en 64 -ee 1 -et 100 -Kway 2 -Qway 8
conda deactivate

echo copying output
module load rclone
rclone copy $SCRDIR/training box:"/transferchpc/lr1 1e-4 fe 100 lds 1 ldr 1 k 5 ds Calcination bse true mag 10000 50000 10000 adam true p true en 64 ee 1 -et 100 Kway 2 Qway 8"
