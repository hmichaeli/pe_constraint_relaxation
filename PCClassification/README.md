# ModelNet40 Classification
Code adapted from the [FlyingGiraffe/vnn-pc](https://github.com/FlyingGiraffe/vnn-pc/tree/master/vn-pointnet) repository

## Data
 Download and unzip the ModelNet40 dataset from [here](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip), and pass its location with `--data_dir`.

## Results Reproduction
All runs use the SO(3)/SO(3) protocol with 300 points and 250 epochs. We report the best validation instance accuracy of the strictly equivariant model, i.e. with the dense relaxation layers or the PE removed (`Best (Equiv) Instance Accuracy` in the VN-PointNet log, `Test (Equiv)` in the VN-DGCNN log). Repeat each command with different `--seed` values to average over seeds, and give every run its own output directory (`--log_dir` for VN-PointNet, `--exp_name` for VN-DGCNN).

### VN-PointNet

```
cd vn-pointnet
```

Baseline
```
python train_cls.py --epoch 250 --num_point 300 --model_mode pointnet_equi --model pointnet_cls_dual --strict_equiv --learning_rate 0.001 --pooling mean --data_dir <DATA_DIR> --batch_size 32 --rot so3 --seed 0 --log_dir vnp_baseline_seed0
```

Constraint Relaxation (CR)
```
python train_cls.py --epoch 250 --num_point 300 --model_mode pointnet_equi --model pointnet_cls_dual --learning_rate 0.001 --pooling mean --data_dir <DATA_DIR> --batch_size 32 --rot so3 --mix_coef_schedule triangular --reg_lambda 0.01 --seed 0 --log_dir vnp_cr_seed0
```

PE-additive (Ours)
```
python train_cls.py --epoch 250 --num_point 300 --model_mode pointnet_equi --model pointnet_poe_cls_dual --learning_rate 0.001 --pooling mean --data_dir <DATA_DIR> --batch_size 32 --rot so3 --poe sinusoidal --mix_coef_schedule linear --reg_lambda 0.001 --seed 0 --log_dir vnp_pe_add_seed0
```

PE-concat (Ours)
```
python train_cls.py --epoch 250 --num_point 300 --model_mode pointnet_equi --model pointnet_poe_cls_dual --learning_rate 0.001 --pooling mean --data_dir <DATA_DIR> --batch_size 32 --rot so3 --poe sinusoidal --poe_concat --mix_coef_schedule linear --reg_lambda 0.005 --seed 0 --log_dir vnp_pe_concat_seed0
```

Outputs are written to `log/classification/<log_dir>/`. Training starts from scratch unless `--resume <epoch>` is given.

### VN-DGCNN
```
cd vn-dgcnn
```

Baseline
```
python main_cls.py --epochs 250 --num_points 300 --model eqcnn --strict_equiv --batch_size 32 --rot so3 --optimizer sgd --data_dir <DATA_DIR> --seed 1 --exp_name vnd_baseline_seed1
```

Constraint Relaxation (CR)
```
python main_cls.py --epochs 250 --num_points 300 --model eqcnn --mix_coef_schedule triangular --regularizer 0.01 --batch_size 32 --rot so3 --optimizer sgd --data_dir <DATA_DIR> --seed 1 --exp_name vnd_cr_seed1
```

PE-additive (Ours)
```
python main_cls.py --epochs 250 --num_points 300 --model eqcnn_poe --batch_size 32 --rot so3 --optimizer sgd --data_dir <DATA_DIR> --poe fixed --mix_coef_schedule linear --regularizer 0.001 --seed 1 --exp_name vnd_pe_add_seed1
```

PE-concat (Ours)
```
python main_cls.py --epochs 250 --num_points 300 --model eqcnn_poe --batch_size 32 --rot so3 --optimizer sgd --data_dir <DATA_DIR> --poe sinusoidal --poe_concat --mix_coef_schedule triangular --regularizer 0.001 --seed 1 --exp_name vnd_pe_concat_seed1
```

Outputs are written to `<results_dir>/results/cls/<exp_name>/` (`--results_dir` defaults to the current directory).
