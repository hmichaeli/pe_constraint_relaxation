#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Congyue Deng
@Contact: congyue@stanford.edu
@File: main.py
"""


from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import ModelNet40

from model_equi import EQCNN_cls,EQCNN_cls_normDual
from eqcnn_poe import EQCNN_POE_cls_normDual
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
from tqdm import tqdm
import torch.nn.functional as F
from pytorch3d.transforms import RotateAxisAngle, Rotate, random_rotations
import wandb


def _init_():
    exp_dir = os.path.join(args.results_dir, 'results/cls', args.exp_name)
    model_dir = os.path.join(exp_dir, 'models')
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    os.system(f'cp main_cls.py {exp_dir}/main.py.backup')
    os.system(f'cp model_equi.py {exp_dir}/model.py.backup')
    os.system(f'cp util.py {exp_dir}/util.py.backup')
    os.system(f'cp data.py {exp_dir}/data.py.backup')

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def train(args, io):
    # Initialize Weights & Biases
    _wb_project = args.wandb_project or os.environ.get('WANDB_PROJECT', 'point_cloud_equivariant')
    _wb_group = args.wandb_group
    wandb.init(
        project=_wb_project,
        name=(args.wandb_name or args.exp_name),
        group=_wb_group,
        config=vars(args)
    )


    train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points,base_dir=args.data_dir), num_workers=8,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points,base_dir=args.data_dir), num_workers=8,
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    if args.model == 'dgcnn':
        model = DGCNN_cls(args).to(device)
    elif args.model == 'eqcnn':
        model = EQCNN_cls_normDual(args).to(device)
    elif args.model == 'eqcnn_poe':
        model = EQCNN_POE_cls_normDual(args).to(device)
    else:
        raise Exception("Not implemented")
    print(str(model))

    model = nn.DataParallel(model)

    print("Let's use", torch.cuda.device_count(), "GPUs!")      
    if args.optimizer.lower() == 'sgd':
        opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
    elif args.optimizer.lower() == 'adam':
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)


    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.min_lr)
    n_parameters = sum(p.numel() for p in model.parameters())
    print("model: ", model)
    print("num of parameters: ", n_parameters)
    print("optimizer: ", opt)
    print("scheduler: ", scheduler)
    print("Using device:", device)
    wandb.run.summary["n_parameters"] = n_parameters

    resume_epoch=0
    if args.resume_epoch>=0:
        checkpoint=torch.load(os.path.join(args.results_dir, 'results/cls', args.exp_name, f'models/model_{args.resume_epoch}.t7'))
        model.load_state_dict(checkpoint['model'])
        opt.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        resume_epoch=args.resume_epoch+1
        print("Resuming from epoch",resume_epoch)

    criterion = cal_loss
    cross_criterion=torch.nn.CrossEntropyLoss() 
    best_test_acc = 0

    # store best test accuracy for logging
    best_acc_instance_equiv = 0.0
    best_acc_class_equiv = 0.0
    best_acc_instance_noproj = 0.0
    best_acc_class_noproj = 0.0
            

    for epoch in range(resume_epoch,args.epochs):
        scheduler.step()

        ####################
        # Train
        ####################
        train_loss = 0.0
        train_norm=0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        if args.mix_coef_schedule=='constant':
            mix_coef=0.1
        elif args.mix_coef_schedule=='none':
            # no scheduled decay: lambda = 1 throughout training
            mix_coef=1.0
        elif args.mix_coef_schedule=='triangular':
            if epoch<=args.epochs/2:
                mix_coef=2*epoch/args.epochs
            else:
                mix_coef=1-2*(epoch-args.epochs/2)/args.epochs
        elif args.mix_coef_schedule=='linear':
            mix_coef = 1 - epoch/args.epochs
        else:
            raise Exception("Mix coef schedule Not implemented")
        
        print("Mix coef",mix_coef)
        cnt=0
        # accumulators for epoch metrics
        reg_ld_sqrt_sum = 0.0
        reg_w_sqrt_sum = 0.0
        batch_count = 0
        for data, label in tqdm(train_loader):
            trot = None
            if args.rot == 'z':
                trot = RotateAxisAngle(angle=torch.rand(data.shape[0])*360, axis="Z", degrees=True, device=device)
            elif args.rot == 'so3':
                trot = Rotate(R=random_rotations(data.shape[0]), device=device)
            
            data, label = data.to(device), label.to(device).squeeze()
            if trot is not None:
                data = trot.transform_points(data)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            
            opt.zero_grad()
            

            logits,n = model(data, args.strict_equiv, mix_coef)
            ld_sqrt = torch.sqrt(n[0] + 1e-8)
            w_sqrt = torch.sqrt(n[1] + 1e-8)
            loss = criterion(logits, label)+mix_coef*args.regularizer*(ld_sqrt+w_sqrt)
            loss.backward()
            
            opt.step()
            cnt+=1
            preds = logits.max(dim=1)[1]
            count += batch_size

            train_loss += loss.item() * batch_size
            #print(train_loss*1.0/count)
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
            # accumulate regularization terms
            reg_ld_sqrt_sum += float(ld_sqrt.item())
            reg_w_sqrt_sum += float(w_sqrt.item())
            batch_count += 1

        del logits

        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        train_loss_avg = train_loss*1.0/count if count > 0 else 0.0
        reg_ld_sqrt_avg = reg_ld_sqrt_sum / max(batch_count, 1)
        reg_w_sqrt_avg = reg_w_sqrt_sum / max(batch_count, 1)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                 train_loss_avg,
                                                                                 metrics.accuracy_score(
                                                                                     train_true, train_pred),
                                                                                 metrics.balanced_accuracy_score(
                                                                                     train_true, train_pred))
        io.cprint(outstr)

        ####################
        # Test
        ####################
        # Validation (no projection / relaxed: equiv=False (strict_equiv))
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        with torch.no_grad():
            for data, label in test_loader:
                trot = None
                if args.rot == 'z':
                    trot = RotateAxisAngle(angle=torch.rand(data.shape[0])*360, axis="Z", degrees=True, device=device)
                elif args.rot == 'so3':
                    trot = Rotate(R=random_rotations(data.shape[0]), device=device)
                data, label = data.to(device), label.to(device).squeeze()
                if trot is not None:
                    data = trot.transform_points(data)
                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]
                # logits,_ = model(data,False,mix_coef)
                logits,_ = model(data, args.strict_equiv, mix_coef)
                
                loss = criterion(logits, label)
                preds = logits.max(dim=1)[1]
                count += batch_size
                test_loss += loss.item() * batch_size
                test_true.append(label.cpu().numpy())
                test_pred.append(preds.detach().cpu().numpy())
            test_true = np.concatenate(test_true)
            test_pred = np.concatenate(test_pred)
            test_acc_np = metrics.accuracy_score(test_true, test_pred)
            avg_per_class_acc_np = metrics.balanced_accuracy_score(test_true, test_pred)
            outstr = 'Test (NoProj) %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                                test_loss*1.0/count if count>0 else 0.0,
                                                                                test_acc_np,
                                                                                avg_per_class_acc_np)
        del logits

        io.cprint(outstr)

        # Validation (equivariant projection: equiv=True)
        test_loss_eq = 0.0
        count_eq = 0.0
        test_pred_eq = []
        test_true_eq = []
        with torch.no_grad():
            for data, label in test_loader:
                trot = None
                if args.rot == 'z':
                    trot = RotateAxisAngle(angle=torch.rand(data.shape[0])*360, axis="Z", degrees=True, device=device)
                elif args.rot == 'so3':
                    trot = Rotate(R=random_rotations(data.shape[0]), device=device)
                data, label = data.to(device), label.to(device).squeeze()
                if trot is not None:
                    data = trot.transform_points(data)
                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]
                logits,_ = model(data,True,mix_coef)
                loss = criterion(logits, label)
                preds = logits.max(dim=1)[1]
                count_eq += batch_size
                test_loss_eq += loss.item() * batch_size
                test_true_eq.append(label.cpu().numpy())
                test_pred_eq.append(preds.detach().cpu().numpy())
            test_true_eq = np.concatenate(test_true_eq)
            test_pred_eq = np.concatenate(test_pred_eq)
            test_acc_eq = metrics.accuracy_score(test_true_eq, test_pred_eq)
            avg_per_class_acc_eq = metrics.balanced_accuracy_score(test_true_eq, test_pred_eq)
            outstr_eq = 'Test (Equiv) %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                                test_loss_eq*1.0/count_eq if count_eq>0 else 0.0,
                                                                                test_acc_eq,
                                                                                avg_per_class_acc_eq)
        io.cprint(outstr_eq)

        # Log to Weights & Biases
        try:
            current_lr = opt.param_groups[0]['lr']
        except Exception:
            current_lr = None

        best_acc_instance_equiv = max(best_acc_instance_equiv, test_acc_eq)
        best_acc_class_equiv = max(best_acc_class_equiv, avg_per_class_acc_eq)
        best_acc_instance_noproj = max(best_acc_instance_noproj, test_acc_np)
        best_acc_class_noproj = max(best_acc_class_noproj, avg_per_class_acc_np)


        wandb.log({
            'epoch': epoch + 1,
            'mix_coef': mix_coef,
            'lr': current_lr,
            'train/loss_avg': train_loss_avg,
            'train/acc_instance': metrics.accuracy_score(train_true, train_pred),
            'train/acc_class': metrics.balanced_accuracy_score(train_true, train_pred),
            'train/reg_ld_sqrt_avg': reg_ld_sqrt_avg,
            'train/reg_w_sqrt_avg': reg_w_sqrt_avg,
            'val/acc_instance_equiv': test_acc_eq,
            'val/acc_class_equiv': avg_per_class_acc_eq,
            'val/acc_instance_noproj': test_acc_np,
            'val/acc_class_noproj': avg_per_class_acc_np,
            'val_best/acc_instance_equiv': best_acc_instance_equiv,
            'val_best/acc_class_equiv': best_acc_class_equiv,
            'val_best/acc_instance_noproj': best_acc_instance_noproj,
            'val_best/acc_class_noproj': best_acc_class_noproj
        })
        
        save_dir = os.path.join(args.results_dir, 'results/cls', args.exp_name, 'models')
        # Save latest
        torch.save({'model':model.state_dict(),
            'optimizer':opt.state_dict(),
            'scheduler':scheduler.state_dict()}, os.path.join(save_dir, 'model_latest.t7'))
        
        # Save periodic
        if args.save_interval > 0 and (epoch + 1) % args.save_interval == 0:
            torch.save({'model':model.state_dict(),
                'optimizer':opt.state_dict(),
                'scheduler':scheduler.state_dict()}, os.path.join(save_dir, f'model_{epoch}.t7'))

        if test_acc_np >= best_test_acc:
            best_test_acc = test_acc_np
            torch.save(model.state_dict(), os.path.join(save_dir, 'model_best.t7'))

    # Finish W&B run
    try:
        wandb.finish()
    except Exception:
        pass

   


def test(args, io):
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points),
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    if args.model == 'dgcnn':
        model = DGCNN_cls(args).to(device)
    elif args.model == 'eqcnn':
        model = EQCNN_cls(args).to(device)
    else:
        raise Exception("Not implemented")
    print(str(model))
    nparameters = sum(p.numel() for p in model.parameters())
    print('Total number of parameters: %d' % nparameters)
    model = nn.DataParallel(model)
    if args.model_path is not None:
        model_path = args.model_path
    else:
        model_path = os.path.join(args.results_dir, 'results/cls', args.exp_name, 'models/model_best.t7')
    model.load_state_dict(torch.load(model_path))
    model = model.eval()
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []
    
    #trot = RotateAxisAngle(angle=120, axis="Y", degrees=True, device=device)
    for data, label in test_loader:
        
        trot = None
        if args.rot == 'z':
            trot = RotateAxisAngle(angle=torch.rand(data.shape[0])*360, axis="Z", degrees=True, device=device)
        elif args.rot == 'so3':
            trot = Rotate(R=random_rotations(data.shape[0]), device=device)

        data, label = data.to(device), label.to(device).squeeze()
        if trot is not None:
            data = trot.transform_points(data)
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        logits = model(data,True)
        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    io.cprint(outstr)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='dgcnn_vnn', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='eqcnn', metavar='N',
                        choices=['dgcnn', 'eqcnn', 'eqcnn_poe'],
                        help='Model to use, [dgcnn, eqcnn, eqcnn_poe]')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training [default: Adam]')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1, 0.1 if using sgd)')
    parser.add_argument('--min-lr', type=float, default=0.001, metavar='LR',
                        help='minimum learning rate for cosine annealing (default: 0.001)')
    parser.add_argument('--mix_thresh',type=float,default=0.2)
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=300,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--regularizer',type=float,default=0.0005)
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default=None, metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--rot', type=str, default='aligned', metavar='N',
                        choices=['aligned', 'z', 'so3'],
                        help='Rotation augmentation to input data')
    # parser.add_argument('--cycle', action='store_true')
    parser.add_argument('--mix_coef_schedule', type=str, default='triangular', choices=['linear', 'triangular', 'none'], 
                        help='type of mix coef schedule; none keeps lambda=1 [default: triangular]')
    parser.add_argument('--pooling', type=str, default='mean', metavar='N',
                        choices=['mean', 'max'],
                        help='VNN only: pooling method.')
    parser.add_argument('--resume_epoch', type=int,default=-1)
    parser.add_argument('--data_dir',type=str,default='modelnet')
    # Strict equivariant training toggle (vanilla VNN without relaxation)
    parser.add_argument('--strict_equiv', action='store_true', default=False,
                        help='If set, trains with exact equivariance (no relaxation) by passing equiv=True during training')
    # Weights & Biases configuration
    parser.add_argument('--wandb_project', type=str, default=None,
                        help='Weights & Biases project name (overrides WANDB_PROJECT env)')
    parser.add_argument('--wandb_group', type=str, default=None,
                        help='Weights & Biases group name for grouping related runs')
    parser.add_argument('--wandb_name', type=str, default=None,
                        help='Weights & Biases run name (overrides default name)')
    parser.add_argument('--poe', type=str, default='sinusoidal', choices=['none', 'sinusoidal', 'fixed', 'fixed_so'],
                        help='Type of position of encoding to use for POE')
    parser.add_argument('--poe_concat', action='store_true', default=False, help='If set, use concatenative POE instead of additive POE')
    parser.add_argument('--poe_scale_init', type=float, default=1.0, help='Initial value of the PE scale gamma')
    parser.add_argument('--poe_learnable_scale', type=str2bool, default=True, help='If False, keep the PE scale gamma fixed at --poe_scale_init')

    parser.add_argument('--results_dir', type=str, default='.',
                        help='Directory to save results')
    parser.add_argument('--save_interval', type=int, default=-1,
                        help='Save checkpoint every n epochs. Default -1 (disable)')
    args = parser.parse_args()

    _init_()

    io = IOStream(os.path.join(args.results_dir, 'results/cls', args.exp_name, 'run.log'))
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)
