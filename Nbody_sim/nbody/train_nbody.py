import argparse
import numpy as np
import torch
import os
import wandb
from torch import nn, optim
from models.segnn.segnn import LinearAppr
from models.segnn.segnn_poe import POEAppr
import time
from e3nn.o3 import Irreps
import e3nn.o3 as o3
from torch_geometric.data import Data
from e3nn.o3 import Irreps, spherical_harmonics
from torch_scatter import scatter


from nbody.dataset_nbody import NBodyDataset

time_exp_dic = {'time': 0, 'counter': 0}


#torch.manual_seed(19)

class O3Transform:
    def __init__(self, lmax_attr):
        self.attr_irreps = Irreps.spherical_harmonics(lmax_attr)

    def __call__(self, graph):
        pos = graph.pos
        vel = graph.vel
        charges = graph.charges

        prod_charges = charges[graph.edge_index[0]] * charges[graph.edge_index[1]]
        rel_pos = pos[graph.edge_index[0]] - pos[graph.edge_index[1]]
        edge_dist = torch.sqrt(rel_pos.pow(2).sum(1, keepdims=True))

        graph.edge_attr = spherical_harmonics(self.attr_irreps, rel_pos, normalize=True, normalization='integral')
        vel_embedding = spherical_harmonics(self.attr_irreps, vel, normalize=True, normalization='integral')
        graph.node_attr = scatter(graph.edge_attr, graph.edge_index[1], dim=0, reduce="mean") + vel_embedding

        vel_abs = torch.sqrt(vel.pow(2).sum(1, keepdims=True))
        mean_pos = pos.mean(1, keepdims=True)

        graph.x = torch.cat((pos - mean_pos, vel, vel_abs), 1)
        graph.additional_message_features = torch.cat((edge_dist, prod_charges), dim=-1)
        return graph


def train(gpu, model, args):
    if args.gpus == 0:
        device = 'cpu'
    else:
        device = torch.device('cuda:' + str(gpu))

    model = model.to(device)

    dataset_train = NBodyDataset(partition='train', dataset_name=args.nbody_name,
                                 max_samples=args.max_samples)
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True)

    dataset_val = NBodyDataset(partition='val', dataset_name=args.nbody_name)
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, drop_last=False)

    dataset_test = NBodyDataset(partition='test', dataset_name=args.nbody_name)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=False)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_mse = nn.MSELoss()
    transform = O3Transform(args.lmax_attr)

    if args.log and gpu == 0:
        if args.wandb_project:
            wandb.init(project=args.wandb_project, group=args.exp_name,
                       name=f"{args.exp_name}_{args.seed}",
                        config=args)
        else:
            if args.time_exp:
                wandb.init(project="Nbody time Jan26", group=args.exp_name,
                        name=f"{args.exp_name}_{args.seed}",
                            config=args)
            else:
                wandb.init(project="SEGNN Nbody Jan26", group=args.exp_name,
                            name=f"{args.exp_name}_{args.seed}",
                            config=args)

    best_val_loss = 1e8
    best_test_loss = 1e8
    best_epoch = 0
    activation = {}
    def get_activation():
        def hook(model, input, output):
            activation['ld_norm'] += output[1]
            activation['norm'] += output[2]
        return hook
    for module in model.modules():
        if isinstance(module, LinearAppr) or isinstance(module, POEAppr):
            module.register_forward_hook(get_activation())
    start_epoch=0
    if args.resume>=0:
        overall_save_dir=os.path.join(args.save_dir,args.exp_name)
        os.makedirs(overall_save_dir, exist_ok=True)
        overall_save_dir=os.path.join(overall_save_dir,f'model_epoch_{args.resume}.pth')
        checkpoint=torch.load(overall_save_dir)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch=args.resume
    # print("Cycle:",args.cycle)
    print("Stict equiv: ",args.strict_equiv)
    print("Mix coef schedule:",args.mix_coef_schedule)
    for epoch in range(start_epoch, args.epochs):
        if args.strict_equiv:
            mix_coef=0.0
        else:
            mix_coef=0.1
            # if args.cycle:
            #     if epoch<=args.epochs/2:
            #             mix_coef=2*epoch/args.epochs
            #     else:
            #             mix_coef=1-2*(epoch-args.epochs/2)/args.epochs

            if args.mix_coef_schedule == 'triangular':
                if epoch <= args.epochs / 2:
                    mix_coef = 2 * epoch / args.epochs
                else:
                    mix_coef = 1 - 2 * (epoch - args.epochs / 2) / args.epochs
            elif args.mix_coef_schedule == 'linear':
                mix_coef = 1 - epoch / args.epochs
        print("Mix coef",mix_coef)
        for module in model.modules():
            if isinstance(module, LinearAppr) or isinstance(module, POEAppr):
                    # module.equiv=False
                    # module.mix_coef=mix_coef
                    #  use strict equiv arg
                    module.equiv=args.strict_equiv
                    module.mix_coef=mix_coef

        train_loss, reg_loss = run_epoch(model, optimizer, loss_mse, epoch, loader_train, transform, device,activation, args)
        if args.log and gpu == 0:
            wandb.log({"Epoch": epoch,
                       "Train mix_coef": mix_coef,
                        "Train MSE": train_loss,
                        "Train Reg Loss": reg_loss,
                        "LR": optimizer.param_groups[0]['lr']})
        for module in model.modules():
            if isinstance(module, LinearAppr) or isinstance(module, POEAppr):
                # module.equiv=False
                # module.mix_coef=0
                #  Always eval equivariant model
                module.equiv=True
                module.mix_coef=0

        if epoch % args.test_interval == 0 or epoch == args.epochs-1:
            #train(epoch, loader_train, backprop=False)
            val_loss, _ = run_epoch(model, optimizer, loss_mse, epoch, loader_val, transform, device,activation, args, backprop=False)
            test_loss, _ = run_epoch(model, optimizer, loss_mse, epoch, loader_test,
                                  transform, device,activation, args, backprop=False)
            if args.log and gpu == 0:
                wandb.log({"Epoch": epoch,
                            "Val MSE": val_loss,
                            "Best Val MSE": best_val_loss})
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_test_loss = test_loss
                best_epoch = epoch

                # save best model
                overall_save_dir=os.path.join(args.save_dir,args.exp_name)
                os.makedirs(overall_save_dir, exist_ok=True)
                overall_save_dir=os.path.join(overall_save_dir,f'model_best.pth')
                torch.save({'model':model.state_dict(),
                        'optimizer':optimizer.state_dict()
                        },overall_save_dir)


            print("*** Best Val Loss: %.5f \t Best Test Loss: %.5f \t Best epoch %d" %
                  (best_val_loss, best_test_loss, best_epoch))
            overall_save_dir=os.path.join(args.save_dir,args.exp_name)
            os.makedirs(overall_save_dir, exist_ok=True)
            overall_save_dir=os.path.join(overall_save_dir,f'model_last.pth')
            torch.save({'model':model.state_dict(),
                        'optimizer':optimizer.state_dict()
                        },overall_save_dir)
    if args.log and gpu == 0:
        wandb.log({"final/Best Val MSE": best_val_loss,
                "final/Test MSE": best_test_loss})
    return best_val_loss, best_test_loss, best_epoch


def run_epoch(model, optimizer, criterion, epoch, loader, transform, device,activation, args, backprop=True):
    if backprop:
        model.train()
    else:
        model.eval()

    res = {'epoch': epoch, 'loss': 0, 'counter': 0, 'reg_loss': 0}
    n_nodes = 5
    batch_size = args.batch_size

    edges = loader.dataset.get_edges(args.batch_size, n_nodes)
    edges = [edges[0], edges[1]]
    edge_index = torch.stack(edges)
    

    for batch_idx, data in enumerate(loader):
        data = [d.to(device) for d in data]
        data = [d.view(-1, d.size(2)) for d in data]
        loc, vel, edge_attr, charges, loc_end = data
        random_rotation=o3.rand_matrix(1).squeeze(0).to('cuda')
        loc=loc@random_rotation
        vel=vel@random_rotation
        loc_end=loc_end@random_rotation
        optimizer.zero_grad()

        if args.time_exp:
            torch.cuda.synchronize()
            t1 = time.time()

        if args.model == 'segnn' or args.model == 'segnn_poe' or args.model == 'seconv':
            graph = Data(edge_index=edge_index, pos=loc, vel=vel, charges=charges, y=loc_end)
            batch = torch.arange(0, batch_size)
            graph.batch = batch.repeat_interleave(n_nodes).long()
            graph = graph.to(device)
            graph = transform(graph)  # Add O3 attributes
            activation['ld_norm']=0
            activation['norm']=0
            loc_pred = graph.pos + model(graph)
        else:
            raise Exception("Unknown model")

        if args.time_exp:
            torch.cuda.synchronize()
            t2 = time.time()
            time_exp_dic['time'] += t2 - t1
            time_exp_dic['counter'] += 1

            if epoch % 100 == 99:
                print("Forward average time: %.6f" % (time_exp_dic['time'] / time_exp_dic['counter']))
                if args.log:
                    wandb.log({"Time": time_exp_dic['time']/time_exp_dic['counter']})
        or_loss = criterion(loc_pred, loc_end)
        reg_loss = args.regul*(torch.sqrt(activation['ld_norm'] + 1e-8)+torch.sqrt(activation['norm'] + 1e-8))
        loss=or_loss + reg_loss
        if backprop:
            loss.backward()
            optimizer.step()
        res['loss'] += or_loss.item()*batch_size
        res['reg_loss'] += reg_loss.item()*batch_size
        res['counter'] += batch_size

    if not backprop:
        prefix = "==> "
    else:
        prefix = ""
    print('%s epoch %d avg loss: %.5f' % (prefix+loader.dataset.partition, epoch, res['loss'] / res['counter']))

    return res['loss'] / res['counter'], res['reg_loss'] / res['counter']
