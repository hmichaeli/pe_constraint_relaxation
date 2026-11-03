# Smoke Plume (C4 Rotation)
Code adapted from the [Rose-STL-Lab/Approximately-Equivariant-Nets](https://github.com/Rose-STL-Lab/Approximately-Equivariant-Nets) repository (Wang et al., ICML 2022, commit `e8e9355`).

Our additions:
- `models/e2cnn_pe.py`: `E2CNN_PE`, the official E2CNN with a learned Fourier positional encoding added to the input, x + λ(t)·γ·PE.
- `run_rotation.py`: training and evaluation on the rotation smoke-plume dataset with the official data splits and training protocol.

All other files are the official code. The only change is in `models/__init__.py`, which imports `E2CNN_PE` and drops a stale import of a commented-out class.

## Dependencies
We ran these experiments with Python 3.10, PyTorch 2.7.0 (CUDA 12.8) and e2cnn 0.2.3.
```bash
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install e2cnn==0.2.3 numpy wandb
```
`requirements.txt` lists the original requirements of the official repository.

## Data
Download the preprocessed smoke-plume datasets from the link in the [official repository](https://github.com/Rose-STL-Lab/Approximately-Equivariant-Nets#dataset-and-preprocessing), and place the rotation files (`raw_data_<task>_<orientation>.pt`) in `PhiFlow/Rotation/`, or pass their location with `--data_dir`.
Alternatively, generate them with `data_gen.ipynb` and PhiFlow 2.0.1, as described there.

## Results Reproduction
Each configuration is run with seeds 0, 1 and 2. Give every run its own `--output_dir`, since the checkpoint name depends only on the model and seed. Training uses the official early stopping, so runs end after about 100–110 epochs (1–2 hours on a single GPU).

E2CNN (strict equivariance)
```
python run_rotation.py --model Equiv --strict_equiv --alpha 0 --mix_coef_schedule none --pe_regularization_lambda 0 --hidden_dim 92 --num_layers 5 --input_length 10 --out_length 6 --batch_size 16 --num_epoch 1000 --learning_rate 0.001 --decay_rate 0.95 --weight_decay 0.0004 --group_order 4 --seed 0 --output_dir results/e2cnn_seed0
```

RSteer (Wang et al.)
```
python run_rotation.py --model RSteer --alpha 0.00001 --restore_best_validation --hidden_dim 92 --num_layers 5 --input_length 10 --out_length 6 --batch_size 16 --num_epoch 1000 --learning_rate 0.001 --decay_rate 0.95 --weight_decay 0.0004 --group_order 4 --seed 0 --output_dir results/rsteer_seed0
```

E2CNN + PE, decayed (Ours)
```
python run_rotation.py --model Equiv --alpha 0 --mix_coef_schedule linear --schedule_max_epoch 50 --pe_regularization_lambda 0.005 --pe_scale_init 1.0 --hidden_dim 92 --num_layers 5 --input_length 10 --out_length 6 --batch_size 16 --num_epoch 1000 --learning_rate 0.001 --decay_rate 0.95 --weight_decay 0.0004 --group_order 4 --seed 0 --output_dir results/e2cnn_pe_decayed_seed0
```

E2CNN + PE, persistent
```
python run_rotation.py --model Equiv --alpha 0 --mix_coef_schedule none --pe_regularization_lambda 0 --pe_scale_init 1.0 --hidden_dim 92 --num_layers 5 --input_length 10 --out_length 6 --batch_size 16 --num_epoch 1000 --learning_rate 0.001 --decay_rate 0.95 --weight_decay 0.0004 --group_order 4 --seed 0 --output_dir results/e2cnn_pe_persistent_seed0
```

The decayed-PE configuration (linear schedule over 50 epochs, β = 0.005) was selected by validation RMSE from a seed-0 sweep over schedule ∈ {linear, triangular}, schedule length ∈ {50, 100} epochs and β ∈ {0.1, 0.05, 0.01, 0.005, 0.001}.

### Reported metrics
All rows use the checkpoint with the lowest validation RMSE. The test metrics are the 20-step rollout RMSE on the Future and Domain test sets, printed at the end of training and stored in `<output_dir>/PhiFlow_Rotation_<model>_seed<seed>.pt`.

| Model | Printed line | Checkpoint keys |
|---|---|---|
| E2CNN | `Best PE-off checkpoint` | `best_pe_off/future_rmse`, `best_pe_off/domain_rmse` |
| RSteer | `Future Test RMSE` | `future_rmse`, `domain_rmse` |
| E2CNN + PE, decayed | `Best PE-off checkpoint` | `best_pe_off/future_rmse`, `best_pe_off/domain_rmse` |
| E2CNN + PE, persistent | `Best PE-on checkpoint` | `best_pe_on/future_rmse`, `best_pe_on/domain_rmse` |

Expected results (mean ± std over seeds 0–2):

| Model | Future RMSE | Domain RMSE |
|---|---|---|
| E2CNN | 0.81 ± 0.01 | 0.60 ± 0.02 |
| RSteer | 0.79 ± 0.00 | 0.58 ± 0.00 |
| E2CNN + PE, decayed (PE removed at test) | 0.80 ± 0.01 | 0.58 ± 0.01 |
| E2CNN + PE, persistent (PE kept at test) | 0.80 ± 0.01 | 0.59 ± 0.01 |

W&B logging is off by default; enable it with `--wandb_mode online --wandb_project <project>`.

## Cite
```
@inproceedings{wang2022approximately,
title={Approximately Equivariant Networks for Imperfectly Symmetric Dynamics},
author={Rui Wang and Robin Walters and Rose Yu},
booktitle={International Conference on Machine Learning},
year={2022},
organization={PMLR}
}
```
