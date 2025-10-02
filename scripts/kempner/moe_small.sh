#!/bin/bash
#SBATCH --job-name=moe_sweep
#SBATCH --account=kempner_grads
#SBATCH --output=/n/netscratch/sham_lab/Lab/pranavajitnair/logs_moe_sweeps/%A-%a.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=72:00:00
#SBATCH --mem=150GB
#SBATCH --partition=kempner_h100
#SBATCH --array=1-2

# --- Environment Setup ---
module load python
source ~/.bashrc
eval "$(mamba shell hook --shell )"
mamba deactivate
mamba activate moe
module load gcc/13.2.0-fasrc01
module load cuda/12.9.1-fasrc01


# export CHECKPOINTS_PATH=/n/netscratch/sham_lab/Lab/pranavajitnair/moe_sweeps/baseline_sweeps/checkpoints/
# export WANDB_MODE=offline
export CHECKPOINTS_PATH=/n/netscratch/sham_lab/Lab/pranavajitnair/moe_sweeps123
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ... other environment variables ...
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH=.:${PYTHONPATH}
export WANDB_API_KEY=491c486a33b983b269e4c64e0a46cd636f347500

# --- Command Execution ---
COMMAND=$(sed -n "${SLURM_ARRAY_TASK_ID}p" /n/netscratch/sham_lab/Lab/pranavajitnair/moe_sweeps_files/sweep_commands.txt)

echo "------------------------------------------------"
echo "SLURM ARRAY TASK ID: ${SLURM_ARRAY_TASK_ID}"
echo "Running command:"
echo "${COMMAND}"
echo "------------------------------------------------"

eval ${COMMAND}
