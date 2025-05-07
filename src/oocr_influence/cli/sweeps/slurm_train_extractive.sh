#!/bin/bash
#SBATCH -p ml
#SBATCH -A ml
#SBATCH -q ml
#SBATCH --cpus-per-task=4
#SBATCH --mem=100GB
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --job-name="influence work"
#SBATCH --nodelist=concerto1,concerto2,concerto3,overture
queue
# Arguments should be a python script and the arguments to pass to it.
#SBATCH -o logs/%A/%A_%a.out
#SBATCH -e logs/%A/%A_%a.err

source ./.venv/bin/activate

# We run the incoming command with the slurm arguments. Command is normally one of the experiment scripts, e.g. python -m oocr_influence.cli.sweeps.learning_rate_experiment
echo $@
$@ --slurm_index $SLURM_ARRAY_TASK_ID --job_id $SLURM_JOB_ID --slurm_array_max_ind $SLURM_ARRAY_TASK_MAX