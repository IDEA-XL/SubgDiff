import os
import shutil
import argparse
import yaml
from easydict import EasyDict
from tqdm.auto import tqdm
from glob import glob
import torch
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from torch_geometric.data import DataLoader




from models.epsnet import get_model
from utils.datasets import ConformationDataset
from utils.transforms import *
from utils.misc import *
from utils.common import get_optimizer, get_scheduler


import torch.distributed as dist
def setup_dist(args, port=None, backend="nccl", verbose=False):
   # TODO
    return rank, local_rank, world_size, device

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt = rt/ nprocs
    return rt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/qm9_500steps.yml')
    parser.add_argument('--device', type=str, default='cuda:4')

    parser.add_argument('--resume_iter', type=int, default=None)
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--distribution', action='store_true', default=False,
                      help='enable ddp running')  
    parser.add_argument('--tag', type=str, default='', help="just for marking the experiments infomation")
    parser.add_argument('--n_jobs', type=int, default=2, help="Dataloader cpu ")
    parser.add_argument('--print_freq', type=int, default=50, help="")
    args = parser.parse_args()

    args.distribution=False # torch.dist

    resume = os.path.isdir(args.config)
    if resume:
        config_path = glob(os.path.join(args.config, '*.yml'))[0]
        resume_from = args.config
    else:
        config_path = args.config

    with open(config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))
    config_name = os.path.basename(config_path)[:os.path.basename(config_path).rfind('.')]
    seed_all(config.train.seed)

    args.local_rank = int(args.device.split(":")[-1])
    if 0 and args.distribution:
        rank, local_rank, world_size, device = setup_dist(args, verbose=True)
        args.device=device
        args.local_rank=local_rank
        setattr(config, 'local_rank', local_rank)
        setattr(config, 'world_size', world_size)
        setattr(config, 'tag', args.tag)


    master_worker = (rank == 0) if  args.distribution else True
    args.nprocs = torch.cuda.device_count()


    if master_worker: 
        # Logging
        if resume:
            log_dir = get_new_log_dir(args.logdir, prefix=config_name+args.tag, tag='resume')
            os.symlink(os.path.realpath(resume_from), os.path.join(log_dir, os.path.basename(resume_from.rstrip("/"))))
        else:
            log_dir = get_new_log_dir(args.logdir, prefix=config_name+args.tag)
            shutil.copytree('./models', os.path.join(log_dir, 'models'))
            shutil.copytree('./utils', os.path.join(log_dir, 'utils'))
        ckpt_dir = os.path.join(log_dir, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        logger = get_logger('train', log_dir)
        writer = torch.utils.tensorboard.SummaryWriter(log_dir)
        logger.info(args)
        logger.info(config)
        shutil.copyfile(config_path, os.path.join(log_dir, os.path.basename(config_path)))

    # Datasets and loaders
    if master_worker: logger.info('Loading datasets...')
    noise_transforms=None
    if config.model.type=='subgraph_diffusion':
        from utils.transforms import SubgraphNoiseTransform
        noise_transforms= SubgraphNoiseTransform(config.model, tag=args.tag)
        # noise_transform_ddpm= SubgraphNoiseTransform(config.model, tag=args.tag, ddpm=False)
        
    transforms = CountNodesPerGraph()
    train_set = ConformationDataset(config.dataset.train, transform=transforms, noise_transform=noise_transforms,config=config.model)
    val_set = ConformationDataset(config.dataset.val, transform=transforms, noise_transform=noise_transforms,config=config.model)
    train_iterator = inf_iterator(DataLoader(train_set, config.train.batch_size, num_workers=args.n_jobs, shuffle=True))
    val_loader = DataLoader(val_set, config.train.batch_size, num_workers=args.n_jobs, shuffle=False)


    if args.distribution:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        train_iterator = inf_iterator(DataLoader(train_set, config.train.batch_size, num_workers=args.n_jobs, shuffle=False,sampler=train_sampler))
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)
        val_loader = DataLoader(val_set, config.train.batch_size, shuffle=False, num_workers=args.n_jobs, sampler=val_sampler)

    # Model
    if master_worker: logger.info('Building model...')
    model = get_model(config.model).to(args.device)
    # model = get_model(config.model).cuda()

    # Optimizer
    optimizer_global = get_optimizer(config.train.optimizer, model.model_global) # module.
    optimizer_local = get_optimizer(config.train.optimizer, model.model_local)
    optimizer_mask = get_optimizer(config.train.optimizer, model.model_mask)
    scheduler_global = get_scheduler(config.train.scheduler, optimizer_global)
    scheduler_local = get_scheduler(config.train.scheduler, optimizer_local)
    scheduler_mask = get_scheduler(config.train.scheduler, optimizer_mask)
    start_iter = 1


    if args.distribution:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    # Resume from checkpoint
    if resume:
        ckpt_path, start_iter = get_checkpoint_path(os.path.join(resume_from, 'checkpoints'), it=args.resume_iter)
        logger.info('Resuming from: %s' % ckpt_path)
        logger.info('Iteration: %d' % start_iter)
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        optimizer_global.load_state_dict(ckpt['optimizer_global'])
        optimizer_local.load_state_dict(ckpt['optimizer_local'])
        try: optimizer_mask.load_state_dict(ckpt['optimizer_mask']) 
        except: pass
        scheduler_global.load_state_dict(ckpt['scheduler_global'])
        scheduler_local.load_state_dict(ckpt['scheduler_local'])
        try: scheduler_mask.load_state_dict(ckpt['scheduler_mask']) 
        except: pass

    def train(it):
        model.train()
        optimizer_global.zero_grad()
        optimizer_local.zero_grad()
        optimizer_mask.zero_grad()
        ddpm_step = config.train.max_iters//2
        batch = next(train_iterator).to(args.device)

        if args.distribution:
            loss_func=model.module.get_loss
        else:
            loss_func=model.get_loss

        loss, loss_global, loss_local, loss_mask = loss_func(
            data=batch,
            atom_type=batch.atom_type,
            pos=batch.pos,
            bond_index=batch.edge_index,
            bond_type=batch.edge_type,
            batch=batch.batch,
            num_nodes_per_graph=batch.num_nodes_per_graph,
            num_graphs=batch.num_graphs,
            anneal_power=config.train.anneal_power,
            return_unreduced_loss=True
        )
        loss_mask = loss_mask.mean()
        if hasattr(batch,"last_select"):
            sum_selected = batch.last_select.sum()
            loss = loss.sum()/sum_selected 
            loss_global=loss_global.sum()/sum_selected 
            loss_local = loss_local.sum()/sum_selected
        else:
            loss = loss.mean()
            loss_global=loss_global.mean()
            loss_local = loss_local.mean()
            
        if args.distribution:

            reduced_loss =reduce_mean(loss, args.nprocs)
            reduced_loss_global = reduce_mean(loss_global, args.nprocs)
            reduced_loss_local = reduce_mean(loss_local, args.nprocs)
            reduced_loss_mask = reduce_mean(loss_mask, args.nprocs)

        loss.backward()
        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
        optimizer_global.step()
        optimizer_local.step()
        optimizer_mask.step()

        if  master_worker and (it-1) % args.print_freq == 0:
            if args.distribution:
                loss=reduced_loss
                loss_global =reduced_loss_global
                loss_local = reduced_loss_local
                loss_mask = reduced_loss_mask
            logger.info('[Train] Epoch %05d | Iter %05d | Loss %.2f | Loss(Global) %.2f | Loss(Local) %.2f | Loss(mask) %.2f | Grad %.2f | LR(Global) %.6f | LR(Local) %.6f| LR(mask) %.6f|%s' % (
            (it*config.train.batch_size)//len(train_set), it, loss.item(), loss_global.item(), loss_local.item(), loss_mask.item(), orig_grad_norm, optimizer_global.param_groups[0]['lr'], 
            optimizer_local.param_groups[0]['lr'],optimizer_mask.param_groups[0]['lr'], log_dir
        ))
            writer.add_scalar('train/loss', loss, it)
            writer.add_scalar('train/loss_global', loss_global.mean(), it)
            writer.add_scalar('train/loss_local', loss_local.mean(), it)
            writer.add_scalar('train/loss_mask', loss_mask.mean(), it)
            writer.add_scalar('train/lr_global', optimizer_global.param_groups[0]['lr'], it)
            writer.add_scalar('train/lr_local', optimizer_local.param_groups[0]['lr'], it)
            writer.add_scalar('train/lr_mask', optimizer_mask.param_groups[0]['lr'], it)
            writer.add_scalar('train/grad_norm', orig_grad_norm, it)
            writer.flush()

    def validate(it):
        sum_loss, sum_n = torch.tensor(0.0).to(args.local_rank), 0
        sum_loss_global, sum_n_global = torch.tensor(0.0).to(args.local_rank), 0
        sum_loss_local, sum_n_local = torch.tensor(0.0).to(args.local_rank), 0
        sum_loss_mask, sum_n_mask = torch.tensor(0.0).to(args.local_rank), 0
        # print("validate....",local_rank,end=' | ')
        with torch.no_grad():
            model.eval()
            for i, batch in enumerate(tqdm(val_loader, desc='Validation',disable=not master_worker)):
                batch = batch.to(args.local_rank)
                if args.distribution:
                    loss_func=model.module.get_loss
                else:
                    loss_func=model.get_loss     

                loss, loss_global, loss_local, loss_mask = loss_func(
                    data=batch,
                    atom_type=batch.atom_type, 
                    pos=batch.pos,
                    bond_index=batch.edge_index,
                    bond_type=batch.edge_type,
                    batch=batch.batch,
                    num_nodes_per_graph=batch.num_nodes_per_graph,
                    num_graphs=batch.num_graphs,
                    anneal_power=config.train.anneal_power,
                    return_unreduced_loss=True
                )
                sum_loss += loss.sum().item()
                sum_loss_local += loss_local.sum().item()
                sum_loss_global += loss_global.sum().item()
                sum_loss_mask += loss_mask.sum().item()
                sum_n_mask += loss_mask.size(0)
                if hasattr(batch, "last_select"):
                    sum_selected = batch.last_select.sum()
                    sum_n +=sum_selected
                    sum_n_local +=sum_selected
                    sum_n_global +=sum_selected
                    
                else:
                    sum_n += loss.size(0)
                    sum_n_local += loss_local.size(0)
                    sum_n_global += loss_global.size(0)
                    
        avg_loss = sum_loss / sum_n
        avg_loss_global = sum_loss_global / sum_n_global
        avg_loss_local = sum_loss_local / sum_n_local
        avg_loss_mask = sum_loss_mask / sum_n_mask

        if args.distribution:
            dist.barrier()
            avg_loss =reduce_mean(avg_loss, args.nprocs)
            avg_loss_global = reduce_mean(avg_loss_global, args.nprocs)
            avg_loss_local = reduce_mean(avg_loss_local, args.nprocs)
            avg_loss_mask = reduce_mean(avg_loss_mask, args.nprocs)
        

        if config.train.scheduler.type == 'plateau':
            scheduler_global.step(avg_loss_global)
            scheduler_local.step(avg_loss_local)
            # scheduler_mask.step(avg_loss_mask)
        else:
            scheduler_global.step()
            scheduler_local.step()
            scheduler_mask.step()

        if master_worker:  
            logger.info('[Validate] Iter %05d | Loss %.6f | Loss(Global) %.6f | Loss(Local) %.6f  | Loss(mask) %.6f' % (
            it, avg_loss, avg_loss_global, avg_loss_local, avg_loss_mask
        ))
            writer.add_scalar('val/loss', avg_loss, it)
            writer.add_scalar('val/loss_global', avg_loss_global, it)
            writer.add_scalar('val/loss_local', avg_loss_local, it)
            writer.add_scalar('val/loss_mask', avg_loss_mask, it)
            writer.flush()
            return avg_loss
        

    if master_worker: print("training....")
    try:
        for it in range(start_iter, config.train.max_iters + 1):
            
            # train_sampler.set_epoch(it)
            train(it)
            # TODO if avg_val_loss < : save
            if it % config.train.val_freq == 0 or it == config.train.max_iters:
                avg_val_loss = validate(it)
                if master_worker and (it % 20000 == 0 or it == config.train.max_iters):
                    # print("saving checkpoint....")
                    ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                    torch.save({
                        'config': config,
                        'model': model.state_dict(),
                        'optimizer_global': optimizer_global.state_dict(),
                        'scheduler_global': scheduler_global.state_dict(),
                        'optimizer_local': optimizer_local.state_dict(),
                        'scheduler_local': scheduler_local.state_dict(),
                        'optimizer_mask': optimizer_mask.state_dict(),
                        'scheduler_mask': scheduler_mask.state_dict(),
                        'iteration': it,
                        'avg_val_loss': avg_val_loss,
                    }, ckpt_path)
    except KeyboardInterrupt:
        if master_worker: logger.info('Terminating...')



    
