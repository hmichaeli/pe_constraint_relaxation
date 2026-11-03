
from data_utils.ModelNetDataLoader import ModelNetDataLoader
import argparse
import numpy as np
import os
import torch
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import sys
import provider
import importlib
from data import ModelNet40
import math
import shutil
from pytorch3d.transforms import RotateAxisAngle, Rotate, random_rotations
import wandb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

#torch.manual_seed(42)
def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training [default: 24]')
    parser.add_argument('--model_mode', default='pointnet_equi', help='equivariant or not [default: pointnet_equi]')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training [default: 200]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training [default: 0.001]')
    parser.add_argument('--gpu', type=str, default=None, help='GPU id to use; if not set, CUDA_VISIBLE_DEVICES is left unchanged [default: None]')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment directory under log/classification/ (or an absolute path); defaults to <model>_<timestamp> [default: None]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate [default: 1e-4]')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    parser.add_argument('--resume', type=int, default=-1,
                        help='Epoch to resume from best_model.pth; negative starts from scratch with zero-initialized dense relaxation layers [default: -1]')
    parser.add_argument('--data_dir',type=str,default='modelnet')
    parser.add_argument('--rot', type=str, default='aligned', metavar='N',
                        choices=['aligned', 'z', 'so3'],
                        help='Rotation augmentation to input data')
    parser.add_argument('--pooling', type=str, default='mean', metavar='N',
                        choices=['mean', 'max'],
                        help='VNN only: pooling method.')
    parser.add_argument('--width', type=int, default=1024,
                        help='Max width (widest layer) to scale channels proportionally [default: 1024]')
    # Strict equivariant training toggle (vanilla VNN without relaxation)
    parser.add_argument('--strict_equiv', action='store_true', default=False,
                        help='If set, trains with exact equivariance (no relaxation) by passing equiv=True during training')
    # Weights & Biases configuration
    parser.add_argument('--wandb_project', type=str, default=None,
                        help='Weights & Biases project name (overrides WANDB_PROJECT env)')
    parser.add_argument('--wandb_group', type=str, default=None,
                        help='Weights & Biases group name for grouping related runs')
    parser.add_argument('--wandb_name', type=str, default=None,
                        help='Weights & Biases run name')
    
    parser.add_argument('--poe', type=str, default=None, help='type of positional encoding [default: None]',
                        choices=[None, 'sinusoidal', 'fixed'])
    parser.add_argument('--poe_concat', action='store_true', default=False, help='If set, use concatenative POE instead of additive POE')
    parser.add_argument('--reg_lambda', type=float, default=0.01, help='Relaxation regularization lambda [default: 0.01]')
    parser.add_argument('--mix_coef_schedule', type=str, default='triangular', choices=['linear', 'triangular'], 
                        help='type of mix coef schedule [default: triangular]')
    parser.add_argument('--seed', type=int, default=0, help='random seed for torch and numpy; negative disables seeding [default: 0]')
    
    return parser.parse_args()

def test(model, loader,mix_coef,equiv, num_class=40):
    mean_correct = []
    class_acc = np.zeros((num_class,3))
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data
        
        trot = None
        if args.rot == 'z':
                trot = RotateAxisAngle(angle=torch.rand(points.shape[0])*360, axis="Z", degrees=True)
        elif args.rot == 'so3':
            trot = Rotate(R=random_rotations(points.shape[0]))
        if trot is not None:
            points = trot.transform_points(points)
        
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        classifier = model.eval()
        
        pred,_,_= classifier(points,equiv,mix_coef)
        pred_choice = pred.data.max(1)[1]
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target==cat].eq(target[target==cat].long().data).cpu().sum()
            class_acc[cat,0]+= classacc.item()/float(points[target==cat].size()[0])
            class_acc[cat,1]+=1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item()/float(points.size()[0]))
    class_acc[:,2] =  class_acc[:,0]/ class_acc[:,1]
    class_acc = np.mean(class_acc[:,2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(parents=True, exist_ok=True)
    experiment_dir = experiment_dir.joinpath('classification')
    experiment_dir.mkdir(parents=True, exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(f'{args.model}_{timestr}')
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(parents=True, exist_ok=True)

    '''LOG'''
    args = parse_args()
    
    if args.seed is not None and args.seed >= 0:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    # Initialize Weights & Biases
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        group=args.wandb_group,
        config=vars(args)
    )


    train_loader = torch.utils.data.DataLoader(ModelNet40(partition='train', num_points=args.num_point,base_dir=args.data_dir), num_workers=8,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(ModelNet40(partition='test', num_points=args.num_point,base_dir=args.data_dir), num_workers=8,
                             batch_size=args.batch_size, shuffle=True, drop_last=False)

    '''MODEL LOADING'''
    num_class = 40
    MODEL = importlib.import_module(args.model_mode + '.' + args.model)
    shutil.copy('./models/%s/%s.py' % (args.model_mode, args.model), str(experiment_dir))
    shutil.copy('./models/%s/pointnet_util.py' % args.model_mode, str(experiment_dir))

    classifier = MODEL.get_model(args, num_class, normal_channel=args.normal).cuda()
    criterion = MODEL.get_loss().cuda()
    n_parameters = sum(p.numel() for p in classifier.parameters())
    print("Classifier:\n", classifier)
    print("Number of classifier parameters: ", n_parameters)
    print("Criterion:\n", criterion)

    wandb.run.summary["model_parameters"] = n_parameters


    # Training starts from scratch unless --resume is set (see below).
    start_epoch = 0


    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    # Track best validation accuracies (equiv and no-proj), similar to vn-dgcnn
    best_acc_instance_equiv = 0.0
    best_acc_class_equiv = 0.0
    best_acc_instance_noproj = 0.0
    best_acc_class_noproj = 0.0
    mean_correct=[]
    if args.resume>=0:
        checkpoint=torch.load( str(checkpoints_dir) + f'/best_model.pth')
        classifier.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch=args.resume
        log_string('Resuming from %s/best_model.pth at epoch %d' % (checkpoints_dir, start_epoch))
    if args.resume<0:
        with torch.no_grad():
            for name, param in classifier.named_parameters():
                
                if ('dense_lin' in name) or ('dense_map_to_dir' in name):

                    param.zero_()
    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch,args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))

        scheduler.step()
        # if epoch<=200/2:
        #     mix_coef=2*epoch/200
        # else:
        #     mix_coef=1-2*(epoch-200/2)/200
        # mix_coef=max(mix_coef,0)
        # print("Mix coef",mix_coef)

        # Bugfix - use num_epochs instead of hardcoded 200
        if args.mix_coef_schedule == 'triangular':
            if epoch <= args.epoch / 2:
                mix_coef = 2 * epoch / args.epoch
            else:
                mix_coef = 1 - 2 * (epoch - args.epoch / 2) / args.epoch
        elif args.mix_coef_schedule == 'linear':
            mix_coef = 1 - epoch / args.epoch
        mix_coef = max(mix_coef, 0)
        print("Mix coef", mix_coef)


        # accumulators for epoch metrics
        train_loss_sum = 0.0
        reg_ld_sqrt_sum = 0.0
        reg_w_sqrt_sum = 0.0
        reg_loss_sum = 0.0
        task_loss_sum = 0.0
        batch_count = 0

        for batch_id, data in tqdm(enumerate(train_loader, 0), total=len(train_loader), smoothing=0.9):
            
            points, target = data

            
            trot = None
            if args.rot == 'z':
                trot = RotateAxisAngle(angle=torch.rand(points.shape[0])*360, axis="Z", degrees=True)
            elif args.rot == 'so3':
                trot = Rotate(R=random_rotations(points.shape[0]))
            if trot is not None:
                points = trot.transform_points(points)
            
            points = points.data.numpy()

            points = provider.random_point_dropout(points)
            points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3])
            points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3])

            points = torch.Tensor(points)
            target = target[:, 0]

            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()

            classifier = classifier.train()

            # Use strict equivariant mode if requested (vanilla VNN)
            pred, trans_feat,n = classifier(points, args.strict_equiv, mix_coef)
            # Add epsilon to sqrt to avoid NaN
            eps = 1e-8
            reg_ld_sqrt = torch.sqrt(n[0] + eps)
            reg_w_sqrt = torch.sqrt(n[1] + eps)
            reg_loss = args.reg_lambda * (reg_ld_sqrt+reg_w_sqrt)
            task_loss = criterion(pred, target.long(), trans_feat) 
            loss = task_loss + reg_loss
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()
            global_step += 1
            # accumulate
            train_loss_sum += float(loss.item())
            reg_ld_sqrt_sum += float(reg_ld_sqrt.item())
            reg_w_sqrt_sum += float(reg_w_sqrt.item())
            reg_loss_sum += float(reg_loss.item())
            task_loss_sum += float(task_loss.item())
            batch_count += 1

        train_instance_acc = np.mean(mean_correct)
        # Compute train averages
        train_loss_avg = train_loss_sum / max(batch_count, 1)
        reg_ld_sqrt_avg = reg_ld_sqrt_sum / max(batch_count, 1)
        reg_w_sqrt_avg = reg_w_sqrt_sum / max(batch_count, 1)

        log_string('Train Instance Accuracy: %f' % train_instance_acc)
        log_string('Train Loss Avg: %.6f | Reg ld sqrt Avg: %.6f | Reg w sqrt Avg: %.6f' % (train_loss_avg, reg_ld_sqrt_avg, reg_w_sqrt_avg))


        with torch.no_grad():
            instance_acc, class_acc = test(classifier.eval(), test_loader,mix_coef=mix_coef,equiv=True)

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            log_string('Test (Equiv) Instance Accuracy: %f, Class Accuracy: %f'% (instance_acc, class_acc))
            log_string('Best (Equiv) Instance Accuracy: %f, Class Accuracy: %f'% (best_instance_acc, best_class_acc))

            if (instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s'% savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler':scheduler.state_dict()
                }
                torch.save(state, savepath)
            
            if (epoch + 1) % 50 == 0 or (epoch + 1) == args.epoch:
                savepath = str(checkpoints_dir) + f'/model_{epoch}.pth'
                log_string('Saving at %s'% savepath)
                state = {
                        'epoch': epoch,
                        'instance_acc': instance_acc,
                        'class_acc': class_acc,
                        'model_state_dict': classifier.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler':scheduler.state_dict()
                    }
                torch.save(state,savepath)
            # Evaluate without projection (NoProj)
            instance_acc_np, class_acc_np = test(classifier.eval(), test_loader,mix_coef=mix_coef,equiv=False)
            log_string('Test (NoProj) Instance Accuracy: %f, Class Accuracy: %f'% (instance_acc_np, class_acc_np))

            # Update best trackers (for W&B logging parity with vn-dgcnn)
            best_acc_instance_equiv = max(best_acc_instance_equiv, instance_acc)
            best_acc_class_equiv = max(best_acc_class_equiv, class_acc)
            best_acc_instance_noproj = max(best_acc_instance_noproj, instance_acc_np)
            best_acc_class_noproj = max(best_acc_class_noproj, class_acc_np)
            log_string('Best So Far -> (Equiv) Inst: %.6f, Class: %.6f | (NoProj) Inst: %.6f, Class: %.6f' % (
                best_acc_instance_equiv, best_acc_class_equiv, best_acc_instance_noproj, best_acc_class_noproj))

            # Log to Weights & Biases
            try:
                current_lr = optimizer.param_groups[0]['lr']
            except Exception:
                current_lr = None
            wandb.log({
                'epoch': epoch + 1,
                'mix_coef': mix_coef,
                'lr': current_lr,
                'train/loss_avg': train_loss_avg,
                'train/acc_instance': train_instance_acc,
                'train/reg_ld_sqrt_avg': reg_ld_sqrt_avg,
                'train/reg_w_sqrt_avg': reg_w_sqrt_avg,
                'train/reg_loss_avg': reg_loss_sum / max(batch_count, 1),
                'train/task_loss_avg': task_loss_sum / max(batch_count, 1),
                'val/acc_instance_equiv': instance_acc,
                'val/acc_class_equiv': class_acc,
                'val/acc_instance_noproj': instance_acc_np,
                'val/acc_class_noproj': class_acc_np,
                'val_best/acc_instance_equiv': best_acc_instance_equiv,
                'val_best/acc_class_equiv': best_acc_class_equiv,
                'val_best/acc_instance_noproj': best_acc_instance_noproj,
                'val_best/acc_class_noproj': best_acc_class_noproj,
            })
            global_epoch += 1

    logger.info('End of training...')
    try:
        wandb.finish()
    except Exception:
        pass
    

if __name__ == '__main__':
    args = parse_args()
    main(args)
