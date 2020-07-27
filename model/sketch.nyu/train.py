from __future__ import division
import os.path as osp
import os
import sys
import time
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from config import config
from dataloader import get_train_loader
from network import Network
from nyu import NYUv2
from utils.init_func import init_weight, group_weight
from engine.lr_policy import WarmUpPolyLR, PolyLR
from engine.engine import Engine
from seg_opr.loss_opr import SigmoidFocalLoss, ProbOhemCrossEntropy2d
from seg_opr.sync_bn import DataParallelModel, Reduce, BatchNorm2d
from tensorboardX import SummaryWriter


try:
    from apex.parallel import DistributedDataParallel, SyncBatchNorm
except ImportError:
    raise ImportError(
        "Please install apex from https://www.github.com/nvidia/apex .")

parser = argparse.ArgumentParser()

port = str(int(float(time.time())) % 20)
os.environ['MASTER_PORT'] = str(10097 + int(port))

with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()

    cudnn.benchmark = True
    seed = config.seed
    if engine.distributed:
        seed = engine.local_rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # data loader
    train_loader, train_sampler = get_train_loader(engine, NYUv2)

    if engine.distributed and (engine.local_rank == 0):
        tb_dir = config.tb_dir + '/{}'.format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
        generate_tb_dir = config.tb_dir + '/tb'
        logger = SummaryWriter(log_dir=tb_dir)
        engine.link_tb(tb_dir, generate_tb_dir)

    # config network and criterion
    criterion = nn.CrossEntropyLoss(reduction='mean',
                                    ignore_index=255)

    if engine.distributed:
        BatchNorm2d = SyncBatchNorm

    model = Network(class_num=config.num_classes, feature=128, bn_momentum=config.bn_momentum,
                    pretrained_model=config.pretrained_model,
                    norm_layer=BatchNorm2d)
    init_weight(model.business_layer, nn.init.kaiming_normal_,
                BatchNorm2d, config.bn_eps, config.bn_momentum,
                mode='fan_in')  # , nonlinearity='relu')

    state_dict = torch.load(config.pretrained_model)  # ['state_dict']
    transformed_state_dict = {}
    for k, v in state_dict.items():
        transformed_state_dict[k.replace('.bn.', '.')] = v

    model.backbone.load_state_dict(transformed_state_dict, strict=False)

    # group weight and config optimizer
    base_lr = config.lr
    if engine.distributed:
        base_lr = config.lr  # * engine.world_size

    ''' fix the weight of resnet'''
    for param in model.backbone.parameters():
        param.requires_grad = False

    params_list = []
    for module in model.business_layer:
        params_list = group_weight(params_list, module, BatchNorm2d,
                                   base_lr )

    optimizer = torch.optim.SGD(params_list,
                                lr=base_lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)

    # config lr policy
    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = PolyLR(base_lr, config.lr_power, total_iteration)

    if engine.distributed:
        print('distributed !!')
        if torch.cuda.is_available():
            model.cuda()
            model = DistributedDataParallel(model)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DataParallelModel(model, device_ids=engine.devices)
        model.to(device)

    engine.register_state(dataloader=train_loader, model=model,
                          optimizer=optimizer)
    if engine.continue_state_object:
        engine.restore_checkpoint()

    model.train()
    print('begin train')

    for epoch in range(engine.state.epoch, config.nepochs):
        if engine.distributed:
            train_sampler.set_epoch(epoch)
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout,
                    bar_format=bar_format)
        dataloader = iter(train_loader)

        sum_loss = 0
        sum_sem = 0
        sum_com = 0
        sum_rest = 0
        sum_sketch = 0
        sum_sketch_gsnn = 0
        sum_kld = 0
        sum_sketch_raw = 0

        for idx in pbar:
            optimizer.zero_grad()
            engine.update_iteration(epoch, idx)

            minibatch = dataloader.next()
            img = minibatch['data']
            hha = minibatch['hha_img']
            label = minibatch['label']
            label_weight = minibatch['label_weight']
            tsdf = minibatch['tsdf']
            depth_mapping_3d = minibatch['depth_mapping_3d']

            sketch_gt = minibatch['sketch_gt']

            img = img.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            hha = hha.cuda(non_blocking=True)
            tsdf = tsdf.cuda(non_blocking=True)
            label_weight = label_weight.cuda(non_blocking=True)
            depth_mapping_3d = depth_mapping_3d.cuda(non_blocking=True)
            sketch_gt = sketch_gt.cuda(non_blocking=True)

            output, _, pred_sketch_raw, pred_sketch_gsnn, pred_sketch, pred_mean, pred_log_var = model(img, depth_mapping_3d, tsdf, sketch_gt)

            cri_weights = torch.FloatTensor([config.empty_loss_weight, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
            criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='none',
                                            weight=cri_weights).cuda()

            '''
            semantic loss
            '''
            selectindex = torch.nonzero(label_weight.view(-1)).view(-1)
            filterLabel = torch.index_select(label.view(-1), 0, selectindex)
            filterOutput = torch.index_select(output.permute(
                0, 2, 3, 4, 1).contiguous().view(-1, 12), 0, selectindex)
            loss_semantic = criterion(filterOutput, filterLabel)
            loss_semantic = torch.mean(loss_semantic)

            '''
            sketch loss
            '''
            filter_sketch_gt = torch.index_select(sketch_gt.view(-1), 0, selectindex)
            filtersketch_raw = torch.index_select(pred_sketch_raw.permute(
                0, 2, 3, 4, 1).contiguous().view(-1, 2), 0, selectindex)
            filtersketch = torch.index_select(pred_sketch.permute(
                0, 2, 3, 4, 1).contiguous().view(-1, 2), 0, selectindex)
            filtersketchGsnn = torch.index_select(pred_sketch_gsnn.permute(
                0, 2, 3, 4, 1).contiguous().view(-1, 2), 0, selectindex)

            criterion_sketch = nn.CrossEntropyLoss(ignore_index=255, reduction='none').cuda()
            loss_sketch = criterion_sketch(filtersketch, filter_sketch_gt)
            loss_sketch = torch.mean(loss_sketch)
            loss_sketch_gsnn = criterion_sketch(filtersketchGsnn, filter_sketch_gt)
            loss_sketch_gsnn = torch.mean(loss_sketch_gsnn)
            loss_sketch_raw = criterion_sketch(filtersketch_raw, filter_sketch_gt)
            loss_sketch_raw = torch.mean(loss_sketch_raw)

            ''' KLD loss '''
            KLD = -0.5 * torch.mean(1 + pred_log_var - pred_mean.pow(2) - pred_log_var.exp())
            
            # reduce the whole loss over multi-gpu
            if engine.distributed:
                dist.all_reduce(loss_semantic, dist.ReduceOp.SUM)
                loss_semantic = loss_semantic / engine.world_size
                dist.all_reduce(loss_sketch, dist.ReduceOp.SUM)
                loss_sketch = loss_sketch / engine.world_size
                dist.all_reduce(loss_sketch_raw, dist.ReduceOp.SUM)
                loss_sketch_raw = loss_sketch_raw / engine.world_size
                dist.all_reduce(loss_sketch_gsnn, dist.ReduceOp.SUM)
                loss_sketch_gsnn = loss_sketch_gsnn / engine.world_size
                dist.all_reduce(KLD, dist.ReduceOp.SUM)
                KLD = KLD / engine.world_size
            else:
                loss = Reduce.apply(*loss) / len(loss)

            current_idx = epoch * config.niters_per_epoch + idx
            lr = lr_policy.get_lr(current_idx)

            optimizer.param_groups[0]['lr'] = lr
            # optimizer.param_groups[1]['lr'] = lr
            for i in range(1, len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr

            loss = loss_semantic \
                   + (loss_sketch+loss_sketch_raw) * config.sketch_weight \
                   + loss_sketch_gsnn * config.sketch_weight_gsnn \
                   + KLD * config.kld_weight
            loss.backward()

            sum_loss += loss.item()
            sum_sem += loss_semantic.item()
            sum_sketch_raw += loss_sketch_raw.item()
            sum_sketch += loss_sketch.item()
            sum_sketch_gsnn += loss_sketch_gsnn.item()
            sum_kld += KLD.item()
            optimizer.step()
            print_str = 'Epoch{}/{}'.format(epoch, config.nepochs) \
                        + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.2e' % lr \
                        + ' loss=%.5f' % (sum_loss / (idx + 1)) \
                        + ' kldloss=%.5f' % (sum_kld / (idx + 1))

            pbar.set_description(print_str, refresh=False)

        if engine.distributed and (engine.local_rank == 0):
            logger.add_scalar('train_loss/tot', sum_loss / len(pbar), epoch)
            logger.add_scalar('train_loss/semantic', sum_sem / len(pbar), epoch)
            logger.add_scalar('train_loss/sketch', sum_sketch / len(pbar), epoch)
            logger.add_scalar('train_loss/sketch_raw', sum_sketch_raw / len(pbar), epoch)
            logger.add_scalar('train_loss/sketch_gsnn', sum_sketch_gsnn / len(pbar), epoch)
            logger.add_scalar('train_loss/KLD', sum_kld / len(pbar), epoch)
            logger.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        if (epoch > config.nepochs // 4) and (epoch % config.snapshot_iter == 0) or (epoch == config.nepochs - 1):
            if engine.distributed and (engine.local_rank == 0):
                engine.save_and_link_checkpoint(config.snapshot_dir,
                                                config.log_dir,
                                                config.log_dir_link)
            elif not engine.distributed:
                engine.save_and_link_checkpoint(config.snapshot_dir,
                                                config.log_dir,
                                                config.log_dir_link)
