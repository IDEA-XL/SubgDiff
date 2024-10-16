import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torch_geometric.loader import DataLoader
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_max_pool, global_mean_pool
from tqdm import tqdm

from finetune.config import args
from finetune.datasets.datasets_QM9 import MoleculeDatasetQM9

from finetune.splitters import qm9_random_customized_01, qm9_random_customized_02, random_scaffold_split, random_split, scaffold_split


def mean_absolute_error(pred, target):
    return np.mean(np.abs(pred - target))


def preprocess_input(one_hot, charges, charge_power, charge_scale):
    charge_tensor = (charges.unsqueeze(-1) / charge_scale).pow(
        torch.arange(charge_power + 1.0, device=device, dtype=torch.float32)
    )  # (-1, 3)
    charge_tensor = charge_tensor.view(charges.shape + (1, charge_power + 1))
    atom_scalars = (
        one_hot.unsqueeze(-1) * charge_tensor
    )  # (N, charge_scale, charge_power + 1)
    atom_scalars = atom_scalars.view(
        charges.shape[:1] + (-1,)
    )  # (N, charge_scale * (charge_power + 1) )
    return atom_scalars


class RandomRotation(object):
    def __init__(self):
        pass

    def __call__(self, x):
        M = np.random.randn(3, 3)
        Q, __ = np.linalg.qr(M)
        return (x @ Q).float()


def split(dataset, data_root):
    if args.split == "scaffold":
        smiles_list = pd.read_csv(data_root + "/processed/smiles.csv", header=None)[
            0
        ].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(
            dataset,
            smiles_list,
            null_value=0,
            frac_train=0.8,
            frac_valid=0.1,
            frac_test=0.1,
        )
        logger.info("scaffold")
    elif args.split == "random":
        train_dataset, valid_dataset, test_dataset = random_split(
            dataset,
            null_value=0,
            frac_train=0.8,
            frac_valid=0.1,
            frac_test=0.1,
            seed=args.seed,
        )
        logger.info("random")
    elif args.split == "random_scaffold":
        smiles_list = pd.read_csv(data_root + "/processed/smiles.csv", header=None)[
            0
        ].tolist()
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(
            dataset,
            smiles_list,
            null_value=0,
            frac_train=0.8,
            frac_valid=0.1,
            frac_test=0.1,
            seed=args.seed,
        )
        logger.info("random scaffold")
    elif args.split == "customized_01" and "qm9" in args.dataset:
        train_dataset, valid_dataset, test_dataset = qm9_random_customized_01(
            dataset, null_value=0, seed=args.seed
        )
        logger.info("customized random on QM9")
    elif args.split == "customized_02" and "qm9" in args.dataset:
        train_dataset, valid_dataset, test_dataset = qm9_random_customized_02(
            dataset, null_value=0, seed=args.seed
        )
        logger.info("customized random (02) on QM9")
    else:
        raise ValueError("Invalid split option on {}.".format(args.dataset))
    logger.info(f"{len(train_dataset)} \t {len(valid_dataset)} \t {len(test_dataset)}")
    return train_dataset, valid_dataset, test_dataset


def model_setup(ckpt=None):
    if args.model_3d=="mask_diff":
        # from models.epsnet import get_model
        from models.epsnet.dualencoder import DualEncoderEpsNetwork
        model = DualEncoderEpsNetwork(ckpt['config'].model).to(args.device)
        graph_pred_linear = torch.nn.Linear(ckpt['config'].model.hidden_dim * 2, num_tasks)

    elif args.model_3d == "schnet":
        from models.epsnet.dualencoder import DualEncoderEpsNetwork,GraphPooling
        model = DualEncoderEpsNetwork(ckpt['config'].model).to(args.device)
        # act= ShiftedSoftplus()
        graph_pred_linear = GraphPooling(ckpt['config'].model.hidden_dim, num_tasks,readout=args.readout).to(args.device)

    elif args.model_3d == "painn":
        from Geom3D.models import  PaiNN
        model = PaiNN(
            n_atom_basis=args.emb_dim,  # default is 64
            n_interactions=args.painn_n_interactions,
            n_rbf=args.painn_n_rbf,
            cutoff=args.painn_radius_cutoff,
            max_z=node_class,
            n_out=num_tasks,
            readout=args.painn_readout,
        )
        graph_pred_linear = model.create_output_layers()

    else:
        raise Exception("3D model {} not included.".format(args.model_3d))
    return model, graph_pred_linear


def load_model(model, graph_pred_linear, model_weight_file):
    logger.info("Loading from {}".format(model_weight_file))
    model_weight = torch.load(model_weight_file, map_location=torch.device('cpu'))
    model.load_state_dict(model_weight["model"])
    if (graph_pred_linear is not None) and ("graph_pred_linear" in model_weight):
        graph_pred_linear.load_state_dict(model_weight["graph_pred_linear"])
    return


def save_model(save_best):
    if not args.output_model_dir == "":
        if save_best:
            output_model_path = os.path.join(args.output_model_dir, "model.pth")
            logger.info(f"save model with optimal loss: {output_model_path}" )
            saved_model_dict = {}
            saved_model_dict["model"] = model.state_dict()
            if graph_pred_linear is not None:
                saved_model_dict["graph_pred_linear"] = graph_pred_linear.state_dict()
            torch.save(saved_model_dict, output_model_path)

        else:
            output_model_path = os.path.join(args.output_model_dir, "model_final.pth")
            logger.info(f"save model in the last epoch: {output_model_path}")
            saved_model_dict = {}
            saved_model_dict["model"] = model.state_dict()
            if graph_pred_linear is not None:
                saved_model_dict["graph_pred_linear"] = graph_pred_linear.state_dict()
            torch.save(saved_model_dict, output_model_path)
    return


def train(epoch, device, loader, optimizer):
    model.train()
    if graph_pred_linear is not None:
        graph_pred_linear.train()

    loss_acc = 0
    num_iters = len(loader)

    if args.verbose:
        L = tqdm(loader)
    else:
        L = loader
    for step, batch in enumerate(L):
        batch = batch.to(device)
        if args.model_3d == "mask_diff":
            edge_inv_global, edge_inv_local, edge_index, edge_type, edge_length, local_edge_mask, node_mask_pred= model(
                            atom_type = batch.atom_type,
                            pos = batch.pos,
                            bond_index = batch.edge_index,
                            bond_type = batch.edge_type,
                            batch = batch.batch,
                            time_step = None,
                            return_edges = True,
                            extend_order = False,
                            extend_radius = True,
                            is_sidechain = None
                        )   # (E_global, 1), (E_local, 1)
            molecule_3D_repr= model.get_graph_embeddings()
            if graph_pred_linear is not None:
                pred = graph_pred_linear(molecule_3D_repr).squeeze()
            else:
                pred = molecule_3D_repr.squeeze()           
        elif args.model_3d == "schnet":
            # molecule_3D_repr = model(batch.x[:, 0], batch.positions, batch.batch)
            edge_inv_global, edge_inv_local, edge_index, edge_type, edge_length, local_edge_mask, node_mask_pred= model(
                            atom_type = batch.atom_type,
                            pos = batch.pos,
                            bond_index = batch.edge_index,
                            bond_type = batch.edge_type,
                            batch = batch.batch,
                            time_step = None,
                            return_edges = True,
                            extend_order = False,
                            extend_radius = True,
                            is_sidechain = None
                        )   # (E_global, 1), (E_local, 1)
            molecule_3D_repr= model.get_global_node_embeddings()
            if graph_pred_linear is not None:
                pred = graph_pred_linear(molecule_3D_repr, batch=batch.batch).squeeze()
            else:
                pred = molecule_3D_repr.squeeze()
        elif args.model_3d == "dimenet":
            molecule_3D_repr = model(batch.x[:, 0], batch.positions, batch.batch)

        elif args.model_3d == "dimenetPP":
            molecule_3D_repr = model(batch.x[:, 0], batch.positions, batch.batch)

        elif args.model_3d == "tfn":
            x_one_hot = F.one_hot(batch.x[:, 0], num_classes=node_class).float()
            x = torch.cat([x_one_hot, batch.x[:, 0].unsqueeze(1)], dim=1).float()
            edge_attr_one_hot = F.one_hot(batch.edge_attr[:, 0], num_classes=edge_class)
            node_3D_repr = model(
                x=x,
                positions=batch.positions,
                edge_index=batch.edge_index,
                edge_feat=edge_attr_one_hot,
            )
            molecule_3D_repr = global_max_pool(node_3D_repr, batch.batch)

        elif args.model_3d == "se3_transformer":
            x_one_hot = F.one_hot(batch.x[:, 0], num_classes=node_class).float()
            x = torch.cat([x_one_hot, batch.x[:, 0].unsqueeze(1)], dim=1).float()
            edge_attr_one_hot = F.one_hot(batch.edge_attr[:, 0], num_classes=edge_class)
            node_3D_repr = model(
                x=x,
                positions=batch.positions,
                edge_index=batch.edge_index,
                edge_feat=edge_attr_one_hot,
            )
            molecule_3D_repr = global_max_pool(node_3D_repr, batch.batch)

        elif args.model_3d == "egnn":
            x_one_hot = F.one_hot(batch.x[:, 0], num_classes=node_class)
            x = preprocess_input(
                x_one_hot,
                batch.x[:, 0],
                charge_power=args.egnn_charge_power,
                charge_scale=node_class,
            )
            node_3D_repr = model(
                x=x,
                positions=batch.positions,
                edge_index=batch.full_edge_index,
                edge_attr=None,
            )
            molecule_3D_repr = global_mean_pool(node_3D_repr, batch.batch)

        elif args.model_3d == "spherenet":
            molecule_3D_repr = model(batch.x[:, 0], batch.positions, batch.batch)

        elif args.model_3d == "segnn":
            molecule_3D_repr = model(batch)

        elif args.model_3d == "painn":
            molecule_3D_repr = model(batch.x, batch.positions, batch.radius_edge_index, batch.batch)

        elif args.model_3d in ["nequip", "allegro"]:
            batch.x = batch.x[:, 0:1]
            batch = batch.to('cuda')
            # TODO: will check how edge_index is constructured.
            data = {
                "edge_index": batch.radius_edge_index,
                "pos": batch.positions,
                "atom_types": batch.x,
                "batch": batch.batch,
            }
            out = model(data)
            molecule_3D_repr = out["total_energy"].squeeze()



        B = pred.size()[0]
        y = batch.y.view(B, -1)[:, task_id]
        # normalize
        y = (y - TRAIN_mean) / TRAIN_std

        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_acc += loss.cpu().detach().item()

        if args.lr_scheduler in ["CosineAnnealingWarmRestarts"]:
            lr_scheduler.step(epoch - 1 + step / num_iters)

    loss_acc /= len(loader)
    if args.lr_scheduler in ["StepLR", "CosineAnnealingLR"]:
        lr_scheduler.step()
    elif args.lr_scheduler in [ "ReduceLROnPlateau"]:
        lr_scheduler.step(loss_acc)

    return loss_acc


@torch.no_grad()
def eval(device, loader):
    model.eval()
    if graph_pred_linear is not None:
        graph_pred_linear.eval()
    y_true = []
    y_scores = []

    if args.verbose:
        L = tqdm(loader)
    else:
        L = loader
    for batch in L:
        batch = batch.to(device)
        if args.model_3d=="mask_diff":
            edge_inv_global, edge_inv_local, edge_index, edge_type, edge_length, local_edge_mask, node_mask_pred= model(
                atom_type = batch.atom_type,
                pos = batch.pos,
                bond_index = batch.edge_index,
                bond_type = batch.edge_type,
                batch = batch.batch,
                time_step = None,
                return_edges = True,
                extend_order = False,
                extend_radius = True,
                is_sidechain = None
            )   # (E_global, 1), (E_local, 1)
            molecule_3D_repr= model.get_graph_embeddings()
            if graph_pred_linear is not None:
                pred = graph_pred_linear(molecule_3D_repr).squeeze()
            else:
                pred = molecule_3D_repr.squeeze()
        if args.model_3d == "schnet":
            # molecule_3D_repr = model(batch.x[:, 0], batch.positions, batch.batch)
                        # molecule_3D_repr = model(batch.x[:, 0], batch.positions, batch.batch)
            edge_inv_global, edge_inv_local, edge_index, edge_type, edge_length, local_edge_mask, node_mask_pred= model(
                            atom_type = batch.atom_type,
                            pos = batch.pos,
                            bond_index = batch.edge_index,
                            bond_type = batch.edge_type,
                            batch = batch.batch,
                            time_step = None,
                            return_edges = True,
                            extend_order = False,
                            extend_radius = True,
                            is_sidechain = None
                        )   # (E_global, 1), (E_local, 1)
            molecule_3D_repr= model.get_global_node_embeddings()
            if graph_pred_linear is not None:
                pred = graph_pred_linear(molecule_3D_repr, batch=batch.batch).squeeze()
            else:
                pred = molecule_3D_repr.squeeze()

        elif args.model_3d == "dimenet":
            molecule_3D_repr = model(batch.x[:, 0], batch.positions, batch.batch)

        elif args.model_3d == "dimenetPP":
            molecule_3D_repr = model(batch.x[:, 0], batch.positions, batch.batch)

        elif args.model_3d == "tfn":
            x_one_hot = F.one_hot(batch.x[:, 0], num_classes=node_class).float()
            x = torch.cat([x_one_hot, batch.x[:, 0].unsqueeze(1)], dim=1).float()
            edge_attr_one_hot = F.one_hot(
                batch.edge_attr[:, 0], num_classes=edge_class
            )
            node_3D_repr = model(
                x=x,
                positions=batch.positions,
                edge_index=batch.edge_index,
                edge_feat=edge_attr_one_hot,
            )
            molecule_3D_repr = global_max_pool(node_3D_repr, batch.batch)

        elif args.model_3d == "se3_transformer":
            x_one_hot = F.one_hot(batch.x[:, 0], num_classes=node_class).float()
            x = torch.cat([x_one_hot, batch.x[:, 0].unsqueeze(1)], dim=1).float()
            edge_attr_one_hot = F.one_hot(
                batch.edge_attr[:, 0], num_classes=edge_class
            )
            node_3D_repr = model(
                x=x,
                positions=batch.positions,
                edge_index=batch.edge_index,
                edge_feat=edge_attr_one_hot,
            )
            molecule_3D_repr = global_max_pool(node_3D_repr, batch.batch)

        elif args.model_3d == "egnn":
            x_one_hot = F.one_hot(batch.x[:, 0], num_classes=node_class).float()
            x = preprocess_input(
                x_one_hot,
                batch.x[:, 0],
                charge_power=args.egnn_charge_power,
                charge_scale=node_class,
            )
            node_3D_repr = model(
                x=x,
                positions=batch.positions,
                edge_index=batch.full_edge_index,
                edge_attr=None,
            )
            molecule_3D_repr = global_mean_pool(node_3D_repr, batch.batch)

        elif args.model_3d == "spherenet":
            molecule_3D_repr = model(batch.x[:, 0], batch.positions, batch.batch)

        elif args.model_3d == "segnn":
            molecule_3D_repr = model(batch)

        elif args.model_3d == "painn":
            molecule_3D_repr = model(batch.x, batch.positions, batch.radius_edge_index, batch.batch)

        elif args.model_3d in ["nequip", "allegro"]:
            batch.x = batch.x[:, 0:1]
            batch = batch.to('cuda')
            data = {
                "edge_index": batch.radius_edge_index,
                "pos": batch.positions,
                "atom_types": batch.x,
                "batch": batch.batch,
            }
            out = model(data)
            molecule_3D_repr = out["total_energy"].squeeze()

            if graph_pred_linear is not None:
                pred = graph_pred_linear(molecule_3D_repr).squeeze()
            else:
                pred = molecule_3D_repr.squeeze()

        B = pred.size()[0]
        y = batch.y.view(B, -1)[:, task_id]
        # denormalize
        pred = pred * TRAIN_std + TRAIN_mean

        y_true.append(y)
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    mae = mean_absolute_error(y_scores, y_true)
    return mae, y_true, y_scores

import csv
def data_write_csv(filename, data,print_data=True):
    if print_data:
        print(data)
    with open(filename, "a", encoding="utf-8", newline='') as w:
        writer = csv.writer(w)
        writer.writerow(data)

if __name__ == "__main__":
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    args.device =device
    ckpt = torch.load(args.ckpt, map_location=torch.device('cpu'))
    if args.config is None:
        from glob import glob
        config_path = glob(os.path.join(os.path.dirname(os.path.dirname(args.ckpt)), '*.yml'))[0]
    else:
        config_path =args.config
    with open(config_path, 'r') as f:
        from easydict import EasyDict
        import yaml
        config = EasyDict(yaml.safe_load(f))
    # seed_all(config.train.seed)
    log_dir = os.path.dirname(os.path.dirname(args.ckpt))

    # Logging
    from utils.misc import get_logger,get_new_log_dir
    if args.tag==None:
        args.tag = args.task
        if args.random_init: args.tag =  "random_init_" +args.tag
    args.tag = args.model_3d +"_" + args.tag
    output_dir = get_new_log_dir(log_dir, 'qm9_finetune', tag=args.tag)
    logger = get_logger('test', output_dir)
    logger.info(args)
    args.output_model_dir = output_dir
    logger.info(f"output_dir:{args.output_model_dir}")
    
    rotation_transform = None
    if args.use_rotation_transform:
        rotation_transform = RandomRotation()

    num_tasks = 1
    assert (args.dataset == "qm9" or args.dataset=="qm9_original")
    data_root = "GeoSSL/molecule_datasets_pyg1.7/{}".format(args.dataset)
    dataset = MoleculeDatasetQM9(
        data_root,
        dataset=args.dataset,
        task=args.task,
        rotation_transform=rotation_transform,
    )
    task_id = dataset.task_id

    ##### Dataset wrapper for graph with radius. #####
    if args.model_3d == "egnn":
        data_root = "../data/molecule_datasets/{}_full".format(args.dataset)
        dataset = MoleculeDataset3DFull(
            data_root,
            preprcessed_dataset=dataset
        )
    elif args.model_3d == "segnn":
        data_root = "../data/molecule_datasets/{}_{}".format(args.dataset, args.segnn_radius)
        dataset = MoleculeDataset3DRadius(
            data_root,
            preprcessed_dataset=dataset,
            radius=args.segnn_radius
        )
    elif args.model_3d in ["painn", "nequip", "allegro"]:
        data_root = "../data/molecule_datasets/{}_{}".format(args.dataset, args.painn_radius_cutoff)
        dataset = MoleculeDataset3DRadius(
            data_root,
            preprcessed_dataset=dataset,
            radius=args.painn_radius_cutoff
        )
    
    if args.only_one_atom_type:
        data_root = "{}_one_atom".format(dataset.root)
        logger.info("neo root", data_root)
        dataset = MoleculeDatasetOneAtom(
            data_root,
            preprcessed_dataset=dataset
        )

    train_dataset, valid_dataset, test_dataset = split(dataset, data_root)
    TRAIN_mean, TRAIN_std = (
        train_dataset.mean()[task_id].item(),
        train_dataset.std()[task_id].item(),
    )
    logger.info("Train mean: {}\tTrain std: {}".format(TRAIN_mean, TRAIN_std))

    if args.loss == "mse":
        criterion = nn.MSELoss()
    elif args.loss == "mae":
        criterion = nn.L1Loss()
    else:
        raise ValueError("Loss {} not included.".format(args.loss))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # set up model
    if args.JK == "concat":
        intermediate_dim = (args.num_layer + 1) * args.emb_dim
    else:
        intermediate_dim = args.emb_dim

    node_class, edge_class = 9, 4

    model, graph_pred_linear = model_setup(ckpt=ckpt)

    if not args.random_init:
        load_model(model, graph_pred_linear, args.ckpt)
    model.to(device)
    # logger.info(model)
    if graph_pred_linear is not None:
        graph_pred_linear.to(device)
    logger.info(graph_pred_linear)

    # set up optimizer
    # different learning rate for different part of GNN
    model_param_group = [{"params": model.parameters(), "lr": args.lr}]
    if graph_pred_linear is not None:
        model_param_group.append(
            {"params": graph_pred_linear.parameters(), "lr": args.lr}
        )
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)

    lr_scheduler = None
    if args.lr_scheduler == "CosineAnnealingLR":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epochs
        )
        logger.info("Apply lr scheduler CosineAnnealingLR")
    elif args.lr_scheduler == "CosineAnnealingWarmRestarts":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, args.epochs, eta_min=1e-4
        )
        logger.info("Apply lr scheduler CosineAnnealingWarmRestarts")
    elif args.lr_scheduler == "StepLR":
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_decay_step_size, gamma=args.lr_decay_factor
        )
        logger.info("Apply lr scheduler StepLR")
    elif args.lr_scheduler == "ReduceLROnPlateau":
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=args.lr_decay_factor, patience=args.lr_decay_patience, min_lr=args.min_lr
        )
        logger.info("Apply lr scheduler ReduceLROnPlateau")
    else:
        logger.info("lr scheduler {} is not included.".format(args.lr_scheduler))

    train_mae_list, val_mae_list, test_mae_list = [], [], []
    best_val_mae, best_val_idx = 1e10, 0
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        loss_acc = train(epoch, device, train_loader, optimizer)
        logger.info("Epoch: {} | Training Loss: {}".format(epoch, loss_acc))

        if epoch % args.print_every_epoch == 0:
            if args.eval_train:
                train_mae, train_target, train_pred = eval(device, train_loader)
            else:
                train_mae = -1
            val_mae, val_target, val_pred = eval(device, val_loader)
            test_mae, test_target, test_pred = eval(device, test_loader)

            train_mae_list.append(train_mae)
            val_mae_list.append(val_mae)
            test_mae_list.append(test_mae)
            logger.info(
                "Evaluation: train: {:.6f}\tval: {:.6f}\ttest: {:.6f}, {}".format(
                    train_mae, val_mae, test_mae, output_dir
                )
            )

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_val_idx = len(train_mae_list) - 1
                if not args.output_model_dir == "":
                    save_model(save_best=True)

                    filename = os.path.join(
                        args.output_model_dir, "evaluation_best.pth"
                    )
                    np.savez(
                        filename,
                        val_target=val_target,
                        val_pred=val_pred,
                        test_target=test_target,
                        test_pred=test_pred,
                    )

                logger.info(
                "[{}] best train: {:.6f}\t best val: {:.6f}\t best test: {:.6f}".format(args.loss,
                    train_mae_list[best_val_idx],
                    val_mae_list[best_val_idx],
                    test_mae_list[best_val_idx],
                ))
        took=time.time() - start_time

        logger.info("Took\t{}\n".format(time.time() - start_time))

    logger.info(
        "best train: {:.6f}\t best val: {:.6f}\t best test: {:.6f}".format(
            train_mae_list[best_val_idx],
            val_mae_list[best_val_idx],
            test_mae_list[best_val_idx],
        )
    )

    save_model(save_best=False)

    args.save_file=log_dir+"/QM9_fintune_results.csv"
    # data_write_csv
    data_write_csv(args.save_file,
                   [
                       f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())},"      
                       f"dataset:{args.dataset}, tag:{args.tag}, model_3d:{args.model_3d} "
                       f" task:{args.task}"] + \
                   [
                       " Best Test:  {:.6f}   best val: {:.6f} best train: {:.6f}".format(test_mae_list[best_val_idx], val_mae_list[best_val_idx], train_mae_list[best_val_idx])
                   ] + [f"seed: {args.seed}, lr_scheduler:{args.lr_scheduler}, batch_size:{args.batch_size}, lr:{args.lr}"
                       f" decay:{args.decay}, best_epoch: {best_val_idx}/{args.epochs},print_every_epoch:{args.print_every_epoch},took:{took}, split:{args.split},ckpt:{args.ckpt} log:{args.output_model_dir}"
                       ])
    print(f"saved successfully in {args.save_file}")