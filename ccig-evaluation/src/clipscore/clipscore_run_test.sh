#!/bin/bash
#SBATCH -D /users/sbsh670/CCIG_Eval/ccig-evaluation/src/clipscore   # Working directory
#SBATCH --job-name clip-clevr-training                      # Job name
#SBATCH --partition=test                       # gpu-a100 Select the correct partition.
#SBATCH --nodes=1                                  # Run on 1 nodes (each node has 48 cores)
#SBATCH --ntasks-per-node=1                        # Run one task
#SBATCH --cpus-per-task=8                          # Use 4 cores, most of the procesing happens on the GPU
#SBATCH --mem=16GB                                 # Expected amount CPU RAM needed (Not GPU Memory)
#SBATCH --time=02:00:00                            # Expected amount of time to run Time limit hrs:min:sec
#SBATCH -e /users/sbsh670/data/ccig-models/clip-clevr/trainFS_/%x_%j.e                         # Standard output and error log [%j is replaced with the jobid]
#SBATCH -o /users/sbsh670/data/ccig-models/clip-clevr/trainFS_/%x_%j.o                         # [%x with the job name], make sure 'results' folder exists.
#Enable modules command

#Remove any unwanted modules
#module purge

#Modules required
#This is an example you need to select the modules your code needs.
#module load python/3.7.12
#module load libs/nvidia-cuda/11.2.0/bin
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/users/sbsh670/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/users/sbsh670/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/users/sbsh670/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/users/sbsh670/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
conda init bash
conda activate clevr-poc
#Run your script.
python clip_clevr_pretraining.py 