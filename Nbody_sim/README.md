# Nbody Simulation
Code adapted from [RobDHess/Steerable-E3-GNN](https://github.com/RobDHess/Steerable-E3-GNN/tree/main) repository.

## Dependencies
Install the dependencies of this experiment through pip by running
```
pip install -r requirements.txt
```

## Creating N-Body data
To recreate the datasets used in this work, navigate to ```nbody/dataset/``` and run
```bash
python3 -u generate_dataset.py --simulation=charged --num-train 10000 --seed 43 --suffix small
```

## Results Reproduction
We report the test MSE at the epoch with the lowest validation MSE (`Best Test Loss` at the end of the log), averaged over 5 runs with different `--seed` values.

Baseline
```
python main.py --batch_size=100 --dataset=nbody --epochs=2000 --gpus=1 --hidden_features=64 --layers=4 --lmax_attr=1 --lmax_h=1 --log=True --max_samples=3000 --model=segnn --norm=none --strict-equiv=True --subspace_type=weightbalanced --lr=0.002 --weight_decay=1e-08
```

Constraint Relaxation (CR)
```
python main.py --batch_size=100 --dataset=nbody --epochs=2000 --gpus=1 --hidden_features=64 --layers=4 --lmax_attr=1 --lmax_h=1 --log=True --max_samples=3000 --model=segnn --norm=none --subspace_type=weightbalanced --lr=0.002 --weight_decay=1e-08 --mix_coef_schedule=linear --regul=0.1
```

PE (Ours)
```
python main.py --batch_size=100 --dataset=nbody --epochs=2000 --gpus=1 --hidden_features=64 --layers=4 --lmax_attr=1 --lmax_h=1 --log=True --max_samples=3000 --model=segnn_poe --norm=none --subspace_type=weightbalanced --lr=0.002 --weight_decay=1e-08 --mix_coef_schedule=triangular --regul=0.05 --poe_scale_init=1 --poe_type=fixed --poe_l1_mask=False
```

