#!/bin/bash
#SBATCH -p general                   ## Use the 'general' partition
#SBATCH -q public                    ## QOS
#SBATCH -c 8                         ## Number of CPU cores
#SBATCH --gres=gpu:a100:1            ## Request 1 A100 GPU (adjust if more GPUs are required)
#SBATCH --mem=64G                    ## Request 64GB of memory (adjust based on model requirements)
#SBATCH --time=24:00:00              ## Request 24 hours of runtime
#SBATCH --job-name=segmentation-finetuning
#SBATCH --output=slurm.%j.out        ## Save stdout to a file with the job ID
#SBATCH --error=slurm.%j.err         ## Save stderr to a file with the job ID
#SBATCH --mail-type=END,FAIL         ## Send email notifications at the end or if the job fails
#SBATCH --mail-user=rjbischo@asu.edu

echo "Starting job in directory: $SLURM_SUBMIT_DIR"
cd $SLURM_SUBMIT_DIR

# Load mamba module
echo "Loading mamba module"
module load mamba/latest

# Activate the existing pytorch environment
echo "Activating the pytorch-gpu-2.3.1-cuda-12.1 environment"
source activate pytorch-gpu-2.3.1-cuda-12.1

# Install dependencies using mamba
echo "Installing PyTorch and torchvision with mamba"
mamba install -c conda-forge pytorch torchvision -y

# Ensure pip installs into the correct environment by using mamba run
echo "Installing segmentation_models_pytorch using pip inside the mamba environment"
mamba run -n pytorch-gpu-2.3.1-cuda-12.1 pip install segmentation-models-pytorch

# Run the Python script for segmentation fine-tuning
echo "Running SegmentationFineTuning.py"
mamba run -n pytorch-gpu-2.3.1-cuda-12.1 python SegmentationFineTuning.py

echo "Job complete"
