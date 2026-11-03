# AF XCiT
Code adapted from [hmichaeli/aft_relaxed_poe](https://github.com/hmichaeli/aft_relaxed_poe) repository.

Experiments were run with PyTorch 2.6 using torchrun for distributed training.

## Data preparation

All experiments were conducted on ImageNet-1k. After extracting the dataset, point `--data-path` at the directory containing `train/` and `val/`. Custom datasets can be integrated by following the interface implemented in `datasets.py`.

## Training
AF-XCIT-NANO + PE (TRAIN AND TEST) 
```
torchrun --nproc_per_node=8 --master_port 1538 main.py --model xcit_af_nano_12_p16 --use-pos true --learnable-pos-scale true --conv-padding-mode circular --pe-down-af true --pe-fuse-down true --pe-act up_gelu --xca-norm-layer layer_af --mlp-act up_gelu --lpi-act up_gelu --features-type cls_attn_af  --data-set IMNET --data-path <DATA-DIR> --batch-size 64  --epochs 400 
```

AF-XCIT-NANO + PE (TRAIN ONLY; OURS) 

```
torchrun --nproc_per_node=8 --master_port 1538 main.py --model xcit_af_nano_12_p16 --use-pos true --learnable-pos-scale true --conv-padding-mode circular --pe-down-af true --pe-fuse-down true --pe-act up_gelu --xca-norm-layer layer_af --mlp-act up_gelu --lpi-act up_gelu --features-type cls_attn_af --pos-decay --pos-decay-steps 800000 --pos-decay-type linear --pos-reg-lambda 0.00001 --data-set IMNET --data-path <DATA-DIR> --batch-size 64  --epochs 400 
```