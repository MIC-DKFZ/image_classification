export CUDA=11.7
export BASE=/dkfz/cluster/gpu/data/OE0441/d246a/conda_envs
export TORCH_CUDA_ARCH_LIST="6.1;7.0;7.5;8.0"

export CONDA_ENV=$BASE/synergy

export PATH=/usr/local/lib:$PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/local/lib:$LIBRARY_PATH
export CPATH=/usr/local/lib:$CPATH
export PATH=/usr/local/cuda-${CUDA}/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-${CUDA}/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-${CUDA}

#module load GCC/14.1.0
#module load binutils/2.42
. ~/.bashrc
conda activate ${CONDA_ENV}

export CUDA_CACHE_DISABLE=1
export OMP_NUM_THREADS=1

export DATASET_LOCATION=/dkfz/cluster/gpu/data/OE0441/s522r/ILSVRC_2012
export EXPERIMENT_LOCATION=/dkfz/cluster/gpu/checkpoints/OE0441/d246a
