# Rapidash - Equivariance Constraint Relaxation

Source: https://openreview.net/forum?id=ghyYc7hgSU

This repository contains the Rapidash models evaluated on the ModelNet40 classification task to study the relaxation of equivariance constraints.

## Running Experiments

Before running the experiments, make sure your datasets are downloaded. You can specify the dataset path using the `--data_dir` argument (e.g., `./data/modelnet40`).

Below are the commands to reproduce our main experiments on ModelNet40:

### 1. Rapidash-Eq (Strict Equivariance)
Strictly SO(3)-equivariant model without pose features.
```bash
python main_modelnet40.py \
    --data_dir ./data/modelnet40 \
    --gpus 1 \
    --num_workers 4 \
    --epochs 120 \
    --warmup 20 \
    --batch_size 32 \
    --lr 1e-4 \
    --weight_decay 1e-8 \
    --hidden_dim '[256,256,256,256]' \
    --basis_dim 256 \
    --basis_hidden_dim 128 \
    --layers '[1,1,1,1]' \
    --orientations 8 \
    --degree 2 \
    --edge_types '["knn-8","knn-8","knn-8","fc"]' \
    --ratios '[0.25,0.25,0.25]' \
    --widening 4 \
    --layer_scale None \
    --equivariance SEn \
    --skip_connections False \
    --scalar_features "[]" \
    --vector_features '["normals"]' \
    --train_augm True \
    --num_points 1024
```

### 2. Rapidash-Symm. break (Pose Conditioned)
Relaxed equivariance using pose features (`vector_features=["normals", "pose"]`).
```bash
python main_modelnet40.py \
    --data_dir ./data/modelnet40 \
    --gpus 1 \
    --num_workers 4 \
    --epochs 120 \
    --warmup 20 \
    --batch_size 32 \
    --lr 1e-4 \
    --weight_decay 1e-8 \
    --hidden_dim '[256,256,256,256]' \
    --basis_dim 256 \
    --basis_hidden_dim 128 \
    --layers '[1,1,1,1]' \
    --orientations 8 \
    --degree 2 \
    --edge_types '["knn-8","knn-8","knn-8","fc"]' \
    --ratios '[0.25,0.25,0.25]' \
    --widening 4 \
    --layer_scale None \
    --equivariance SEn \
    --skip_connections False \
    --scalar_features "[]" \
    --vector_features '["normals", "pose"]' \
    --train_augm True \
    --num_points 1024
```

### 3. Rapidash-Eq + PE (Constraint Relaxation)
Strictly equivariant model trained with positional encodings (PE) that act as an equivariance constraint relaxation.
```bash
python main_modelnet40.py \
    --data_dir ./data/modelnet40 \
    --gpus 1 \
    --num_workers 4 \
    --epochs 120 \
    --warmup 20 \
    --batch_size 32 \
    --lr 1e-4 \
    --weight_decay 1e-8 \
    --hidden_dim '[256,256,256,256]' \
    --basis_dim 256 \
    --basis_hidden_dim 128 \
    --layers '[1,1,1,1]' \
    --orientations 8 \
    --degree 2 \
    --edge_types '["knn-8","knn-8","knn-8","fc"]' \
    --ratios '[0.25,0.25,0.25]' \
    --widening 4 \
    --layer_scale None \
    --equivariance SEn \
    --skip_connections False \
    --scalar_features "[]" \
    --vector_features '["normals"]' \
    --pe_enabled True \
    --pe_mode add \
    --pe_type sinusoidal \
    --pe_schedule linear \
    --pe_gamma_init 1.0 \
    --pe_reg_beta 0.05 \
    --pe_keep_at_test False \
    --train_augm True \
    --num_points 1024
```

### Evaluation under random rotations
To evaluate a trained model on the original test set and on a randomly rotated test set (an independent SO(3) rotation per sample; Rapidash-Eq + PE is evaluated with the PE removed), run:
```bash
python eval_rotation.py --ckpt <output_dir>/checkpoints/<checkpoint>.ckpt
```
Point sampling and rotations are random, so use `--repeats` to average over several passes.
