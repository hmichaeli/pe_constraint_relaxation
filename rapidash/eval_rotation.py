"""Evaluate a trained ModelNet40 checkpoint on the original and on a randomly rotated test set.

Each test sample in the rotated set is transformed by an independent uniformly sampled SO(3)
rotation, applied to both the points and the normals. Models trained with PE (or pose)
relaxation are evaluated as the strictly equivariant model (lambda = 0), i.e. with the
relaxation removed, as in the paper.

Example:
    python eval_rotation.py --ckpt <output_dir>/checkpoints/last.ckpt
"""
import argparse
import json

import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader

from main_modelnet40 import ModelNet40Model
from utils import NormalizeCoord, RandomSOd, SamplePoints


class RandomRotation:
    """Rotate points and normals of a sample by a uniformly sampled SO(3) rotation."""
    def __init__(self):
        self.generator = RandomSOd(3)

    def __call__(self, data):
        rot = self.generator()
        data.pos = data.pos @ rot.T
        if hasattr(data, 'normal'):
            data.normal = data.normal @ rot.T
        return data


@torch.no_grad()
def evaluate(model, loader, device):
    relaxed = model.hparams.pe_enabled or model.hparams.pose_relax_enabled
    correct = total = 0
    for data in loader:
        data = data.to(device)
        pred = model._predict(data, lambda_override=0.0) if relaxed else model(data)
        correct += (pred.argmax(dim=-1) == data.y).sum().item()
        total += data.y.numel()
    return correct / total


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ModelNet40Model.load_from_checkpoint(args.ckpt, map_location=device, weights_only=False)
    model = model.to(device).eval()
    hp = model.hparams
    data_dir = args.data_dir or hp.data_dir

    results = {'ckpt': args.ckpt}
    for mode in ['standard', 'rotated']:
        accs = []
        for r in range(args.repeats):
            torch.manual_seed(args.seed + r)
            transforms = [
                NormalizeCoord(),
                SamplePoints(num=hp.num_points, remove_faces=True, include_normals=True),
            ]
            if mode == 'rotated':
                transforms.append(RandomRotation())
            dataset = ModelNet(data_dir, name='40', train=False, transform=T.Compose(transforms))
            loader = DataLoader(dataset, batch_size=args.batch_size or hp.batch_size,
                                shuffle=False, num_workers=args.num_workers)
            accs.append(evaluate(model, loader, device))
        results[f'{mode}_acc'] = sum(accs) / len(accs)
        results[f'{mode}_acc_per_repeat'] = accs
        print(f"{mode} accuracy: {results[f'{mode}_acc']:.4f}")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ModelNet40 evaluation under random SO(3) rotations')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to a Lightning checkpoint (.ckpt)')
    parser.add_argument('--data_dir', type=str, default=None, help='ModelNet40 directory [default: the one used for training]')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size [default: the one used for training]')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--repeats', type=int, default=1, help='Number of evaluation passes to average (point sampling and rotations are random)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed of the first pass')
    parser.add_argument('--output', type=str, default=None, help='Optional path of a JSON file for the results')
    main(parser.parse_args())
