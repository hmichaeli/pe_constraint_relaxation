# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Modified from: https://github.com/facebookresearch/deit
"""
import os
import json
from typing import Optional
import numpy as np
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform


class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR100':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 100
    elif args.data_set == 'CIFAR100_SUBSET':
        dataset = CIFAR100Subset(args.data_path, train=is_train, transform=transform, download=True,
                                 samples_per_class=args.samples_per_class, seed=args.dataset_seed)
        nb_classes = 100
    elif args.data_set == 'CIFAR10':
        dataset = datasets.CIFAR10(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 10
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'CARS':
        root = os.path.join(args.data_path, 'train' if is_train else 'test')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 196
    elif args.data_set == 'FLOWERS':
        root = os.path.join(args.data_path, 'train' if is_train else 'test')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 102
    
    elif args.data_set == 'TINY_IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 200

    elif args.data_set == 'StanfordCars':
        dataset = datasets.StanfordCars(args.data_path, split=('train' if is_train else 'test'),
                                        transform=transform, download=False)
        nb_classes = 196
    
    elif args.data_set == 'Flowers102':
        dataset = datasets.Flowers102(args.data_path, split=('train' if is_train else 'test'),
                                       transform=transform, download=True)
        nb_classes = 102

    else:
        raise ValueError(f"Unknown dataset: {args.data_set}")


    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)

        return transform

    t = []

    if args.full_crop:
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        transformations = {}
        transformations = transforms.Compose(
            [transforms.Resize(args.input_size, interpolation=3),
                transforms.CenterCrop(args.input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
        return transformations

    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)



class CIFAR100Subset(datasets.CIFAR100):
    """
    A CIFAR-100 dataset that, for the training split, uses only a fixed number of
    samples per class. The test/eval split is kept intact.

    Args:
        root (str): Dataset root directory.
        train (bool): If True, create dataset from training split, else test split.
        transform (callable, optional): Input transform.
        target_transform (callable, optional): Target transform.
        download (bool): If True, downloads the dataset.
        samples_per_class (int, optional): Number of training samples to keep per class.
            If None or not set, uses the full training set. Ignored for test split.
        seed (int, optional): RNG seed for reproducible sampling. Default: 0.
        shuffle_within_class (bool): Shuffle indices within each class before taking
            the first `samples_per_class`. Default: True.
    """
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform=None,
        target_transform=None,
        download: bool = False,
        *,
        samples_per_class: Optional[int] = None,
        seed: int = 0,
        shuffle_within_class: bool = True,
    ):
        super().__init__(
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

        # Only sub-sample the training split
        if self.train and samples_per_class is not None:
            if samples_per_class <= 0:
                raise ValueError("samples_per_class must be a positive integer.")

            targets = np.asarray(self.targets)
            data = self.data
            rng = np.random.default_rng(seed)

            selected_indices = []
            num_classes = 100

            for c in range(num_classes):
                cls_idx = np.where(targets == c)[0]
                if shuffle_within_class:
                    rng.shuffle(cls_idx)
                k = min(samples_per_class, cls_idx.shape[0])
                if k == 0:
                    # CIFAR-100 train split has samples for every class, but guard anyway
                    continue
                selected_indices.append(cls_idx[:k])

            if len(selected_indices) == 0:
                # Should never happen, but keep a helpful error
                raise RuntimeError("No indices selected; check samples_per_class and dataset integrity.")

            selected_indices = np.concatenate(selected_indices, axis=0)
            # Keep deterministic class-balanced ordering if desired:
            # selected_indices = np.sort(selected_indices)

            self.data = data[selected_indices]
            self.targets = targets[selected_indices].tolist()