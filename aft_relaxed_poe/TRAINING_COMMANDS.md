This file lists the official training commands used in the paper. Replace $OUT_DIR and $DATA_PATH with your own output and dataset paths.

Nano baseline

```
torchrun --nproc_per_node=8 --master_port=1534 main.py --model xcit_nano_12_p16 --batch-size 128 --output_dir=$OUT_DIR --data-path $DATA_PATH
```

Nano APS

```
torchrun --nproc_per_node=8 --master_port=1534 main.py --model xcit_aps_nano_12_p16 --use-pos false --batch-size 64 --output_dir=$OUT_DIR --data-path $DATA_PATH
```

Nano AFT

```
torchrun --nproc_per_node=8 --master_port=1534 main.py --model xcit_af_nano_12_p16  --use-pos false --conv-padding-mode circular --pe-down-af true --pe-fuse-down true --pe-act up_gelu --xca-norm-layer layer_af --mlp-act up_gelu --lpi-act up_gelu --features-type cls_attn_af --batch-size 64 --output_dir=$OUT_DIR --data-path $DATA_PATH
```

Small baseline

```
torchrun --nproc_per_node=8 --master_port=1534 main.py --model xcit_small_12_p16 --drop-path 0.05 --batch-size 128 --output_dir=$OUT_DIR --data-path $DATA_PATH
```

Small APS

```
torchrun --nproc_per_node=8 --master_port=1534 main.py --model xcit_aps_small_12_p16 --use-pos false --drop-path 0.05 --batch-size 64 --output_dir=$OUT_DIR --data-path $DATA_PATH
```

Small AFT

```
torchrun --nproc_per_node=8 --master_port=1534 main.py --model xcit_af_small_12_p16  --use-pos false --conv-padding-mode circular --pe-down-af true --pe-fuse-down true --pe-act up_gelu --xca-norm-layer layer_af --mlp-act up_gelu --lpi-act up_gelu --features-type cls_attn_af --drop-path 0.05 --batch-size 64 --output_dir=$OUT_DIR --data-path $DATA_PATH
```

Notes
- These commands assume a single node with 8 GPUs. For different setups adjust --nproc_per_node and master_port accordingly.
- When running on clusters, ensure CUDA_VISIBLE_DEVICES and environment variables are set appropriately or use the included `run_with_submitit.py` helper.
