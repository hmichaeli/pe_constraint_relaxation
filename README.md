# Positional Encoding is All You Need for Scalable Equivariance Constraint Relaxation

The code is based on the following repositories.
* Improving Equivariant Model Training via Constraint Relaxation [link](https://github.com/StefanosPert/Equivariant_Optimization_CR)
* Alias-Free Transformer: Fractional Shift Invariance via Linear Attention [link](https://github.com/hmichaeli/alias_free_vit)
* Approximately Equivariant Networks for Imperfectly Symmetric Dynamics [link](https://github.com/Rose-STL-Lab/Approximately-Equivariant-Nets)



## Environment Setup
Unless noted otherwise, we run all experiments on a single machine with 8 × NVIDIA RTX A6000. 

The smoke_plume experiments use a separate environment (Python 3.10, PyTorch 2.7, e2cnn 0.2.3) and were run on NVIDIA RTX PRO 6000 Blackwell GPUs; see [smoke_plume/README.md](smoke_plume/README.md).

For the PCClassification and Nbody_sim experiments we used the following environment. Our runs used the CUDA 11.8 build of PyTorch 2.6.0 (`torch==2.6.0+cu118`); the commands below install the CUDA 12.4 build of the same version.

```bash
# Create conda environment
conda create -n pc-torch-2-6 python=3.9
conda activate pc-torch-2-6

# Install PyTorch with CUDA 12.4
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Install core dependencies
pip install tqdm h5py wandb e3nn==0.5.8 timm==0.4.12 torch-geometric==2.6.1 scikit-learn

# Install PyTorch3D
pip install --extra-index-url https://miropsota.github.io/torch_packages_builder pytorch3d==0.7.8+5043d15pt2.6.0cu124

# Install geometric packages
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.6.0+cu124.html 
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
```

## Requirements

- Python 3.9
- CUDA 12.4 compatible GPU (or CPU-only by adjusting PyTorch installation)