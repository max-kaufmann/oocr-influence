#!/bin/bash
#SBATCH -p ml
#SBATCH -A ml
#SBATCH -q ml
#SBATCH --cpus-per-task=4
#SBATCH --mem=100GB
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --job-name="influence work"
#SBATCH --nodelist=concerto1,concerto2,concerto3

# Arguments should be a python script and the arguments to pass to it.
#SBATCH -o logs/%A/%A_%a.out
#SBATCH -e logs/%A/%A_%a.err

source ./.venv/bin/activate
python_script=$1
shift

export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "Running $python_script with $@"

python $python_script --slurm_index $SLURM_ARRAY_TASK_ID --job_id $SLURM_JOB_ID --slurm_array_max_ind $SLURM_ARRAY_TASK_MAX $@