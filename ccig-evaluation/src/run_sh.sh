#!/bin/bash
#SBATCH -D /users/sbsh670/CCIG_Eval/ccig-evaluation   # Working directory
#SBATCH --job-name clipscore                 # Job name
#SBATCH --partition=gpu-a100                       # gpu-a100 Select the correct partition.
#SBATCH --nodes=1                                  # Run on 1 nodes (each node has 48 cores)
#SBATCH --ntasks-per-node=1                        # Run one task
#SBATCH --cpus-per-task=4                          # Use 4 cores, most of the procesing happens on the GPU
#SBATCH --mem=24GB                                 # Expected amount CPU RAM needed (Not GPU Memory)
#SBATCH --time=02:30:00                          # Expected amount of time to run Time limit hrs:min:sec
#SBATCH -e /users/sbsh670/data/ccig-models/clip-clevr/Clipscore_/%x_%j.e                         # Standard output and error log [%j is replaced with the jobid]
#SBATCH -o /users/sbsh670/data/ccig-models/clip-clevr/Clipscore_/%x_%j.o                         # [%x with the job name], make sure 'results' folder exists.
#SBATCH --gres=gpu:1                               # Use one gpu or 2
#Enable modules command
source /opt/flight/etc/setup.sh
flight env activate gridware

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



python -m src.run --images-dir /users/sbsh670/data/ccig-generated-images/gpt-image-2-low/clevr_1_scenes_SAT --analysis_out /users/sbsh670/data/ccig-generated-images/gpt-image-2-low/clevr_1_scenes_SAT --prompts-file /users/sbsh670/data/ccig_evalData/clevr_1_scenes_SAT.jsonl --domain clevr --clipscore_results /users/sbsh670/data/ccig-generated-images/gpt-image-2-low/clevr_1_scenes_SAT/clipscore/results-L-gen.json --vlm_judge_results /users/sbsh670/data/ccig-generated-images/gpt-image-2-low/clevr_1_scenes_SAT/vlm_judge/results-gen.json --soft_tifa_results /users/sbsh670/data/ccig-generated-images/gpt-image-2-low/clevr_1_scenes_SAT/soft_tifa/results-gen.json --perception_results /users/sbsh670/data/ccig-generated-images/gpt-image-2-low/clevr_1_scenes_SAT/perception/results-gen.json --human_results /users/sbsh670/data/ccig-generated-images/gpt-image-2-low/clevr_1_scenes_SAT/human_eval/human_eval/results-gen.json --method analysis    

#python -m src.run --images-dir /users/sbsh670/data/ccig-generated-images/gpt-image-2-low/clevr_1_scenes_SAT --prompts-file /users/sbsh670/data/ccig_evalData/clevr_1_scenes_SAT.jsonl --method human-eval --domain clevr --out-dir /users/sbsh670/data/ccig-generated-images/gpt-image-2-low/clevr_1_scenes_SAT/human_eval --annotation_file /users/sbsh670/data/ccig-generated-images/gpt-image-2-low/clevr_1_scenes_SAT/human_eval/clevr_1_scenes_SAT_human-perception.json --manifest /users/sbsh670/data/ccig-generated-images/gpt-image-2-low/clevr_1_scenes_SAT/manifest.jsonl

#python -m src.run --images-dir /users/sbsh670/data/ccig-generated-images/gpt-image-2-low/clevr_1_scenes_UNSAT --prompts-file /users/sbsh670/data/ccig_evalData/clevr_1_scenes_UNSAT.jsonl --method perception --domain clevr --out-dir /users/sbsh670/data/ccig-generated-images/gpt-image-2-low/clevr_1_scenes_UNSAT --detector grounding-dino --attribute-classifier clip-zero-shot --manifest /users/sbsh670/data/ccig-generated-images/gpt-image-2-low/clevr_1_scenes_UNSAT/manifest.jsonl

#python -m src.run --images-dir /users/sbsh670/data/ccig-generated-images/gpt-image-2-low/clevr_1_scenes_SAT --prompts-file /users/sbsh670/data/ccig_evalData/clevr_1_scenes_SAT.jsonl --method soft-tifa --domain clevr --out-dir /users/sbsh670/data/ccig-generated-images/gpt-image-2-low/clevr_1_scenes_SAT --vqa-backend gpt-4o --manifest /users/sbsh670/data/ccig-generated-images/gpt-image-2-low/clevr_1_scenes_SAT/manifest.jsonl --is_closed_model --sat
#python -m src.run --images-dir /users/sbsh670/data/ccig-generated-images/gpt-image-2-low/clevr_1_scenes_UNSAT --prompts-file /users/sbsh670/data/ccig_evalData/clevr_1_scenes_UNSAT.jsonl --method vlm-judge --domain clevr --out-dir /users/sbsh670/data/ccig-generated-images/gpt-image-2-low/clevr_1_scenes_UNSAT --judge-backend gpt-4o --manifest /users/sbsh670/data/ccig-generated-images/gpt-image-2-low/clevr_1_scenes_UNSAT/manifest.jsonl --is_closed_model
#python -m src.run --images-dir /users/sbsh670/data/ccig-generated-images/gpt-image-2-low/clevr_1_scenes_UNSAT --prompts-file /users/sbsh670/data/ccig_evalData/clevr_1_scenes_UNSAT.jsonl --method clipscore --domain clevr --out-dir /users/sbsh670/data/ccig-generated-images/gpt-image-2-low/clevr_1_scenes_UNSAT --clip-checkpoint /users/sbsh670/Long-CLIP/checkpoints/longclip-L.pt --manifest /users/sbsh670/data/ccig-generated-images/gpt-image-2-low/clevr_1_scenes_UNSAT/manifest.jsonl --is_closed_model
#python -m src.run --images-dir /users/sbsh670/data/ccig-generated-images/gpt-image-2-low/clevr_1_scenes_SAT --prompts-file /users/sbsh670/data/ccig_evalData/clevr_1_scenes_SAT.jsonl --method clipscore --domain clevr --out-dir /users/sbsh670/data/ccig-generated-images/gpt-image-2-low/clevr_1_scenes_SAT --clip-checkpoint openai/clip-vit-large-patch14 --manifest /users/sbsh670/data/ccig-generated-images/gpt-image-2-low/clevr_1_scenes_SAT/manifest.jsonl --is_closed_model
