import os
import argparse

import torch
import torchmetrics
import pytorch_lightning as pl
from lightning_fabric.utilities.exceptions import MisconfigurationException
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader

import wandb

from models.rapidash import Rapidash
from models.rapidash_pe import RapidashPE
from models.relaxation import PoseRelaxation
from utils import (CosineWarmupScheduler, NormalizeCoord, RandomJitter,
                   RandomRotatePerturbation, RandomShift, RandomSOd,
                   SamplePoints, TimerCallback)

# Performance optimization
torch.set_float32_matmul_precision('medium')

# Some augmentation functions
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform, RandomScale


class ModelNet40Model(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        
        # Setup rotation augmentation
        self.rotation_generator = RandomSOd(3)
        
        # Calculate total input channels
        in_channels = (
            3 * ("coords" in self.hparams.scalar_features) +  # x,y,z coordinates as scalars
            3 * ("normals" in self.hparams.scalar_features) +  # normal components as scalars
            1 * ("coords" in self.hparams.vector_features) +  # position as vector
            1 * ("normals" in self.hparams.vector_features) +  # normal as vector
            3 * ("pose" in self.hparams.vector_features)      # pose matrix (3 vectors)
        )

        # Ensure at least one input channel if none are specified
        if in_channels == 0:
            in_channels = 1  # will use constant ones as input

        # Initialize model (Table 6 model-8 semantics are preserved by default:
        # vector features include normals + pose/global frame).
        backbone = Rapidash(
            input_dim=in_channels,
            hidden_dim=self.hparams.hidden_dim,
            output_dim=40,  # ModelNet40 has 40 classes
            num_layers=self.hparams.layers,
            edge_types=self.hparams.edge_types,
            equivariance=self.hparams.equivariance,
            ratios=self.hparams.ratios,
            output_dim_vec=0,
            dim=3,
            num_ori=self.hparams.orientations,
            degree=self.hparams.degree,
            widening_factor=self.hparams.widening,
            layer_scale=self.hparams.layer_scale,
            task_level='graph',
            last_feature_conditioning=False,
            skip_connections=self.hparams.skip_connections,
            basis_dim=self.hparams.basis_dim,
            basis_hidden_dim=self.hparams.basis_hidden_dim
        )

        if self.hparams.pe_enabled:
            self.net = RapidashPE(
                backbone=backbone,
                processed_input_dim=in_channels,
                pe_mode=self.hparams.pe_mode,
                pe_type=self.hparams.pe_type,
                pe_schedule=self.hparams.pe_schedule,
                pe_gamma_init=self.hparams.pe_gamma_init,
                pe_keep_at_test=self.hparams.pe_keep_at_test,
                pe_reg_beta=self.hparams.pe_reg_beta,
            )
        else:
            self.net = backbone

        self.pose_relaxation = None
        if self.hparams.pose_relax_enabled:
            self.pose_relaxation = PoseRelaxation(
                enabled=True,
                schedule=self.hparams.pose_relax_schedule,
                gamma_init=self.hparams.pose_gamma_init,
                keep_at_test=self.hparams.pose_keep_at_test,
                reg_beta=self.hparams.pose_reg_beta,
            )
        
        # Setup metrics
        self.train_metric = torchmetrics.Accuracy(task="multiclass", num_classes=40)
        self.valid_metric = torchmetrics.Accuracy(task="multiclass", num_classes=40)
        self.test_metric = torchmetrics.Accuracy(task="multiclass", num_classes=40)

        # When any relaxation is active, also track the relaxed model's accuracy
        # so that both the strict (lambda=0) and the relaxed models can be compared each epoch.
        if self.hparams.pe_enabled or self.hparams.pose_relax_enabled:
            self.valid_metric_relaxed = torchmetrics.Accuracy(task="multiclass", num_classes=40)
            self.test_metric_relaxed = torchmetrics.Accuracy(task="multiclass", num_classes=40)

        self.best_valid_acc = 0.0

    def _step_context(self):
        """Return (global_step, total_steps) for annealing schedule computation.
        Falls back to (0, 1) when the trainer is not yet attached."""
        total_steps = 1
        try:
            estimated = getattr(self.trainer, "estimated_stepping_batches", None)
            if estimated is not None:
                total_steps = max(int(estimated), 1)
        except (RuntimeError, MisconfigurationException):
            # trainer.test(): estimated_stepping_batches needs the fit train
            # dataloader and raises MisconfigurationException (Lightning 2.6).
            pass
        return int(self.global_step), total_steps

    def _current_train_lambda(self, step: int, total_steps: int) -> float:
        """Return the lambda that the active scheduler would produce at this training step."""
        if self.hparams.pe_enabled:
            return self.net.pe_relaxation.scheduler(step, total_steps)
        if self.pose_relaxation is not None:
            return self.pose_relaxation.scheduler(step, total_steps)
        return 1.0

    def _predict(self, data, lambda_override=None):
        """Core forward pass; lambda_override forces a specific annealing coefficient.

        lambda_override=None  → normal schedule (training) or 0/keep_at_test (eval)
        lambda_override=0.0   → strict / projected equivariant model
        lambda_override=value → relaxed model at that specific lambda
        """
        step, total_steps = self._step_context()
        if self.training and self.hparams.train_augm:
            rot = self.rotation_generator().type_as(data.pos)
            data.pos = torch.einsum('ij,bj->bi', rot, data.pos)
            if hasattr(data, 'normal'):
                data.normal = torch.einsum('ij,bj->bi', rot, data.normal)
        else:
            rot = torch.eye(3, device=data.pos.device)

        # Prepare input features
        x = []  # scalar features
        vec = []  # vector features

        # Add scalar features
        if "coords" in self.hparams.scalar_features:
            x.append(data.pos)
        if "normals" in self.hparams.scalar_features and hasattr(data, 'normal'):
            x.append(data.normal)

        # Add vector features
        if "coords" in self.hparams.vector_features:
            vec.append(data.pos[:,None,:])
        if "normals" in self.hparams.vector_features and hasattr(data, 'normal'):
            vec.append(data.normal[:,None,:])
        if "pose" in self.hparams.vector_features:
            pose_feature = rot.transpose(-2, -1).unsqueeze(0).expand(data.pos.shape[0], -1, -1)
            if self.pose_relaxation is not None:
                pose_feature = self.pose_relaxation.apply_pose(
                    pose_feature=pose_feature,
                    step=step,
                    total_steps=total_steps,
                    training=self.training,
                    lambda_override=lambda_override,
                )
            vec.append(pose_feature)

        # Combine features
        if not x and not vec:  # Only add constant ones if both x and vec are empty
            x = torch.ones(data.pos.size(0), 1).type_as(data.pos)
        else:
            x = torch.cat(x, dim=-1) if x else None
        vec = torch.cat(vec, dim=1) if vec else None

        # Forward pass
        if self.hparams.pe_enabled:
            pred, _ = self.net(x, data.pos, data.edge_index, data.batch, vec=vec,
                               step=step, total_steps=total_steps,
                               lambda_override=lambda_override)
        else:
            pred, _ = self.net(x, data.pos, data.edge_index, data.batch, vec=vec)
        return pred

    def forward(self, data):
        """Standard forward used by the Lightning training loop."""
        return self._predict(data)

    def training_step(self, data, batch_idx):
        pred = self(data)
        task_loss = torch.nn.functional.cross_entropy(pred, data.y)
        loss = task_loss

        if self.hparams.pe_enabled:
            pe_reg_loss = self.net.regularization_loss()
            loss = loss + pe_reg_loss
            self.log("train_pe_lambda", self.net.pe_relaxation.last_lambda, prog_bar=False)
            self.log("train_pe_norm", self.net.pe_relaxation.last_pe_norm, prog_bar=False)
            self.log("train_pe_reg_loss", pe_reg_loss, prog_bar=False)

        if self.pose_relaxation is not None and "pose" in self.hparams.vector_features:
            pose_reg_loss = self.pose_relaxation.regularization_loss()
            loss = loss + pose_reg_loss
            self.log("train_pose_lambda", self.pose_relaxation.last_lambda, prog_bar=False)
            self.log("train_pose_norm", self.pose_relaxation.last_pose_norm, prog_bar=False)
            self.log("train_pose_reg_loss", pose_reg_loss, prog_bar=False)

        self.train_metric(pred, data.y)
        self.log("train_task_loss", task_loss, prog_bar=False)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, data, batch_idx):
        if self.hparams.pe_enabled or self.hparams.pose_relax_enabled:
            # Strict / projected equivariant model: lambda=0 (no PE, no pose annealing contribution)
            pred_strict = self._predict(data, lambda_override=0.0)
            self.valid_metric(pred_strict, data.y)

            # Relaxed model: use the same lambda as the scheduler would at this training step
            step, total_steps = self._step_context()
            current_lam = self._current_train_lambda(step, total_steps)
            pred_relaxed = self._predict(data, lambda_override=current_lam)
            self.valid_metric_relaxed(pred_relaxed, data.y)
        else:
            pred = self(data)
            self.valid_metric(pred, data.y)

    def test_step(self, data, batch_idx):
        if self.hparams.pe_enabled or self.hparams.pose_relax_enabled:
            # Strict / projected equivariant model
            pred_strict = self._predict(data, lambda_override=0.0)
            self.test_metric(pred_strict, data.y)

            # Relaxed model at whatever lambda the schedule gives at this step
            step, total_steps = self._step_context()
            current_lam = self._current_train_lambda(step, total_steps)
            pred_relaxed = self._predict(data, lambda_override=current_lam)
            self.test_metric_relaxed(pred_relaxed, data.y)
        else:
            pred = self(data)
            self.test_metric(pred, data.y)

    def on_train_epoch_end(self):
        self.log("train_acc", self.train_metric, prog_bar=True)

    def on_validation_epoch_end(self):
        # valid_acc = strict / projected equivariant model (lambda=0) — used for checkpointing
        valid_acc = self.valid_metric.compute().item()
        if valid_acc > self.best_valid_acc:
            self.best_valid_acc = valid_acc
            
        self.log("valid_acc", self.valid_metric, prog_bar=True)
        self.log("best_valid_acc", self.best_valid_acc, prog_bar=True)
        if self.hparams.pe_enabled or self.hparams.pose_relax_enabled:
            # valid_acc_relaxed = relaxed model at current training lambda — monitors training progress
            self.log("valid_acc_relaxed", self.valid_metric_relaxed, prog_bar=True)

    def on_test_epoch_end(self):
        self.log("test_acc", self.test_metric, prog_bar=True)
        if self.hparams.pe_enabled or self.hparams.pose_relax_enabled:
            self.log("test_acc_relaxed", self.test_metric_relaxed, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = CosineWarmupScheduler(optimizer, self.hparams.warmup, self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

def load_data(args):
    """Load and preprocess ModelNet40 dataset using PyG."""
    
    # Define transforms
    train_transform = T.Compose([
        NormalizeCoord(),
        SamplePoints(num=args.num_points, remove_faces=True, include_normals=True),
        RandomRotatePerturbation(angle_sigma=0.06, angle_clip=0.18),
        RandomScale((0.8, 1.25)),
        RandomShift(shift_range=0.1),
        RandomJitter(sigma=0.01, clip=0.05),
    ])

    test_transform = T.Compose([
        NormalizeCoord(),
        SamplePoints(num=args.num_points, remove_faces=True, include_normals=True),
    ])
    
    # Create datasets
    train_dataset = ModelNet(
        args.data_dir,
        name='40',
        train=True,
        transform=train_transform,
    )
    
    test_dataset = ModelNet(
        args.data_dir,
        name='40',
        train=False,
        transform=test_transform,
    )
   
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    return train_loader, test_loader

def main(args):
    # Set random seed
    pl.seed_everything(args.seed)

    # Create output directories for logs/checkpoints/artifacts.
    output_dir = os.path.abspath(args.output_dir)
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    train_loader, test_loader = load_data(args)

    # Setup hardware configuration
    if args.gpus > 0:
        accelerator = "gpu"
        devices = args.gpus
    else:
        accelerator = "cpu"
        devices = "auto"
        
    # Configure logging
    if args.log:
        wandb_kwargs = {
            "project": args.wandb_project,
            "save_dir": output_dir,
            "name": args.wandb_name,
            "group": args.wandb_group,
            "config": vars(args),
        }
        if args.wandb_entity is not None:
            wandb_kwargs["entity"] = args.wandb_entity

        logger = pl.loggers.WandbLogger(
            **wandb_kwargs
        )
    else:
        logger = None

    # Setup callbacks
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            monitor='valid_acc',
            mode='max',
            dirpath=checkpoints_dir,
            save_last=True
        ),
        TimerCallback()
    ]
    if args.log:
        callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval='epoch'))

    # Initialize trainer
    trainer = pl.Trainer(
        logger=logger,
        default_root_dir=output_dir,
        max_epochs=args.epochs,
        callbacks=callbacks,
        gradient_clip_val=0.5,
        accelerator=accelerator,
        devices=devices,
        enable_progress_bar=args.enable_progress_bar
    )

    # Train or test
    if args.test_ckpt is None:
        model = ModelNet40Model(args)
        trainer.fit(model, train_loader, test_loader)
        # Use last model (not the best) because we cannot use the test set for model selection.
        # weights_only=False: Lightning checkpoints hold NumPy scalars that the
        # PyTorch >= 2.6 weights-only unpickler rejects.
        trainer.test(model, test_loader, ckpt_path=callbacks[0].last_model_path, weights_only=False)
    else:
        model = ModelNet40Model.load_from_checkpoint(args.test_ckpt, weights_only=False)
        trainer.test(model, test_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ModelNet40 Classification Training')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--warmup', type=int, default=10, help='Number of warmup epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-12, help='Weight decay')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    
    # Model architecture
    parser.add_argument('--hidden_dim', type=eval, default=[128,128,128,128,128], help='Hidden dimension(s)')
    parser.add_argument('--basis_dim', type=int, default=256, help='Basis dimension')
    parser.add_argument('--basis_hidden_dim', type=int, default=128, help='Hidden dimension of the basis function MLP')
    parser.add_argument('--layers', type=eval, default=[1,1,1,1,1], help='Layers per scale')
    parser.add_argument('--orientations', type=int, default=8, help='Number of orientations')
    parser.add_argument('--degree', type=int, default=2, help='Polynomial degree')
    parser.add_argument('--edge_types', type=eval, default=["knn-8","knn-8","knn-8","knn-8","fc"], help='Edge types')
    parser.add_argument('--ratios', type=eval, default=[0.25,0.25,0.25,0.25], help='Pooling ratios')
    parser.add_argument('--widening', type=int, default=4, help='Network widening factor')
    parser.add_argument('--layer_scale', type=eval, default=None, help='Layer scaling factor')
    parser.add_argument('--equivariance', type=str, default="SEn", help='Type of equivariance')
    parser.add_argument('--skip_connections', type=eval, default=False, help='Use U-Net style skip connections')
    
    # Input features
    parser.add_argument('--scalar_features', type=eval, default=[], help='Features to use as scalars: ["coords", "normals"]')
    parser.add_argument('--vector_features', type=eval, default=["normals","pose"], help='Features to use as vectors: ["coords", "normals", "pose"]')

    # PE relaxation (train-time symmetry breaking annealed to zero by default at test)
    parser.add_argument('--pe_enabled', type=eval, default=False, help='Enable PE relaxation wrapper')
    parser.add_argument('--pe_mode', type=str, default="add", choices=["add", "concat"], help='PE injection mode')
    parser.add_argument('--pe_type', type=str, default="fixed", choices=["fixed", "sinusoidal", "learned"], help='PE type')
    parser.add_argument('--pe_schedule', type=str, default="linear", choices=["linear", "triangular", "constant"], help='PE annealing schedule')
    parser.add_argument('--pe_reg_beta', type=float, default=0.0, help='PE magnitude regularization weight')
    parser.add_argument('--pe_keep_at_test', type=eval, default=False, help='Keep PE active at test time')
    parser.add_argument('--pe_gamma_init', type=float, default=1.0, help='Initial PE scale gamma')

    # Pose/global-frame annealing (model12->model11 style probe, adapted to ModelNet)
    parser.add_argument('--pose_relax_enabled', type=eval, default=False, help='Anneal pose/global-frame vector feature during training')
    parser.add_argument('--pose_relax_schedule', type=str, default="linear", choices=["linear", "triangular", "constant"], help='Pose annealing schedule')
    parser.add_argument('--pose_gamma_init', type=float, default=1.0, help='Initial pose scale gamma')
    parser.add_argument('--pose_keep_at_test', type=eval, default=False, help='Keep pose feature active at test time when pose annealing is enabled')
    parser.add_argument('--pose_reg_beta', type=float, default=0.0, help='Pose magnitude regularization weight')
    
    # Training features
    parser.add_argument('--train_augm', type=eval, default=True, help='Use rotation augmentation during training')
    parser.add_argument('--num_points', type=int, default=1024, help='Number of points to sample')

    # Sweep configuration
    parser.add_argument('--config', type=eval, default=None, help='Sweep configuration dictionary')
    parser.add_argument('--model_id', type=int, default=None, help='Model ID in case you would want to label the configuration')
    
    # Data and logging
    parser.add_argument('--data_dir', type=str, default="./datasets/modelnet", help='Data directory')
    parser.add_argument('--output_dir', type=str, default="./outputs/modelnet40", help='Directory to save logs, checkpoints, and artifacts')
    parser.add_argument('--log', type=eval, default=True, help='Enable logging')
    parser.add_argument('--wandb_project', type=str, default="Rapidash-ModelNet40", help='Weights & Biases project name')
    parser.add_argument('--wandb_name', type=str, default=None, help='Weights & Biases run name')
    parser.add_argument('--wandb_group', type=str, default=None, help='Weights & Biases group name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='Weights & Biases entity/team name')
    
    # System and checkpointing
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--enable_progress_bar', type=eval, default=True, help='Show progress bar')
    parser.add_argument('--test_ckpt', type=str, default=None, help='Checkpoint for testing')
    parser.add_argument('--resume_ckpt', type=str, default=None, help='Checkpoint to resume from')
    
    args = parser.parse_args()

    # Overwrite default settings with values from config if provided
    if args.config is not None:
        for key, value in args.config.items():
            setattr(args, key, value)
    
    main(args)
