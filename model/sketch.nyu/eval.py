#!/usr/bin/env python3
# encoding: utf-8
import os
import cv2
import argparse
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.multiprocessing as mp

from config import config
from utils.pyt_utils import ensure_dir, link_file, load_model, parse_devices
from utils.visualize import print_iou, show_img
from engine.evaluator import Evaluator
from engine.logger import get_logger
from seg_opr.metric import hist_info, compute_score
from nyu import NYUv2
from network import Network
from dataloader import ValPre
from torch.utils.data import dataloader
from torch.multiprocessing import reductions
from multiprocessing.reduction import ForkingPickler

logger = get_logger()


default_collate_func = dataloader.default_collate


def default_collate_override(batch):
  dataloader._use_shared_memory = False
  return default_collate_func(batch)

setattr(dataloader, 'default_collate', default_collate_override)

for t in torch._storage_classes:
  if sys.version_info[0] == 2:
    if t in ForkingPickler.dispatch:
        del ForkingPickler.dispatch[t]
  else:
    if t in ForkingPickler._extra_reducers:
        del ForkingPickler._extra_reducers[t]


class SegEvaluator(Evaluator):
    def func_per_iteration(self, data, device):
        img = data['data']
        label = data['label']
        hha = data['hha_img']
        tsdf = data['tsdf']
        label_weight = data['label_weight']
        depth_mapping_3d = data['depth_mapping_3d']

        name = data['fn']
        sketch_gt = data['sketch_gt']
        pred, pred_sketch = self.eval_ssc(img, hha, tsdf, depth_mapping_3d, sketch_gt, device)

        results_dict = {'pred':pred, 'label':label, 'label_weight':label_weight,
                        'name':name, 'mapping':depth_mapping_3d}

        if self.save_path is not None:
            ensure_dir(self.save_path)
            ensure_dir(self.save_path+'_sketch')
            fn = name + '.npy'
            np.save(os.path.join(self.save_path, fn), pred)
            np.save(os.path.join(self.save_path+'_sketch', fn), pred_sketch)
            logger.info('Save the pred npz ' + fn)

        return results_dict

    def hist_info(self, n_cl, pred, gt):
        assert (pred.shape == gt.shape)
        k = (gt >= 0) & (gt < n_cl)  # exclude 255
        labeled = np.sum(k)
        correct = np.sum((pred[k] == gt[k]))

        return np.bincount(n_cl * gt[k].astype(int) + pred[k].astype(int),
                           minlength=n_cl ** 2).reshape(n_cl,
                                                        n_cl), correct, labeled

    def compute_metric(self, results):
        hist_ssc = np.zeros((config.num_classes, config.num_classes))
        correct_ssc = 0
        labeled_ssc = 0

        # scene completion
        tp_sc, fp_sc, fn_sc, union_sc, intersection_sc = 0, 0, 0, 0, 0

        for d in results:
            pred = d['pred'].astype(np.int64)
            label = d['label'].astype(np.int64)
            label_weight = d['label_weight'].astype(np.float32)
            mapping = d['mapping'].astype(np.int64).reshape(-1)

            flat_pred = np.ravel(pred)
            flat_label = np.ravel(label)

            nonefree = np.where(label_weight > 0)  # Calculate the SSC metric. Exculde the seen atmosphere and the invalid 255 area
            nonefree_pred = flat_pred[nonefree]
            nonefree_label = flat_label[nonefree]

            h_ssc, c_ssc, l_ssc = self.hist_info(config.num_classes, nonefree_pred, nonefree_label)
            hist_ssc += h_ssc
            correct_ssc += c_ssc
            labeled_ssc += l_ssc

            occluded = (mapping == 307200) & (label_weight > 0) & (flat_label != 255)   # Calculate the SC metric on the occluded area
            occluded_pred = flat_pred[occluded]
            occluded_label = flat_label[occluded]

            tp_occ = ((occluded_label > 0) & (occluded_pred > 0)).astype(np.int8).sum()
            fp_occ = ((occluded_label == 0) & (occluded_pred > 0)).astype(np.int8).sum()
            fn_occ = ((occluded_label > 0) & (occluded_pred == 0)).astype(np.int8).sum()

            union = ((occluded_label > 0) | (occluded_pred > 0)).astype(np.int8).sum()
            intersection = ((occluded_label > 0) & (occluded_pred > 0)).astype(np.int8).sum()

            tp_sc += tp_occ
            fp_sc += fp_occ
            fn_sc += fn_occ
            union_sc += union
            intersection_sc += intersection

        score_ssc = compute_score(hist_ssc, correct_ssc, labeled_ssc)
        IOU_sc = intersection_sc / union_sc
        precision_sc = tp_sc / (tp_sc + fp_sc)
        recall_sc = tp_sc / (tp_sc + fn_sc)
        score_sc = [IOU_sc, precision_sc, recall_sc]

        result_line = self.print_ssc_iou(score_sc, score_ssc)
        return result_line

    def eval_ssc(self, img, disp, tsdf, depth_mapping_3d, sketch_gt, device=None):
        ori_rows, ori_cols, c = img.shape
        input_data, input_disp = self.process_image_rgbd(img, disp, crop_size=None)
        score, bin_score, sketch_score = self.val_func_process_ssc(input_data, input_disp, tsdf, depth_mapping_3d, sketch_gt, device)
        score = score.permute(1, 2, 3, 0)       # h, w, z, c
        sketch_score = sketch_score.permute(1, 2, 3, 0)

        data_output = score.cpu().numpy()
        sketch_output = sketch_score.cpu().numpy()

        pred = data_output.argmax(3)        # 60x36x60
        pred_sketch = sketch_output.argmax(3)

        return pred, pred_sketch

    def val_func_process_ssc(self, input_data, input_disp, tsdf, input_mapping, sketch_gt, device=None):
        input_data = np.ascontiguousarray(input_data[None, :, :, :], dtype=np.float32)
        input_data = torch.FloatTensor(input_data).cuda(device)

        input_disp = np.ascontiguousarray(input_disp[None, :, :, :], dtype=np.float32)
        input_disp = torch.FloatTensor(input_disp).cuda(device)

        # print(input_mapping.shape, 'hhhhhhhh')
        input_mapping = np.ascontiguousarray(input_mapping[None, :], dtype=np.int32)
        input_mapping = torch.LongTensor(input_mapping).cuda(device)

        tsdf = np.ascontiguousarray(tsdf[None, :], dtype=np.float32)
        tsdf = torch.FloatTensor(tsdf).cuda(device)

        with torch.cuda.device(input_data.get_device()):
            self.val_func.eval()
            self.val_func.to(input_data.get_device())
            with torch.no_grad():
                score, bin_score, sketch_score = self.val_func(input_data, input_mapping, tsdf)
                score = score[0]
                sketch_score = sketch_score[0]

                # if self.is_flip:
                #     input_data = input_data.flip(-1)
                #     input_disp = input_disp.flip(-1)
                #     score_flip = self.val_func(input_data, input_disp)
                #     score_flip = score_flip[0]
                #     score += score_flip.flip(-1)
                score = torch.exp(score)

        return score, bin_score, sketch_score

    def print_ssc_iou(self, sc, ssc):
        lines = []
        lines.append('--*-- Semantic Scene Completion --*--')
        lines.append('IOU: \n{}\n'.format(str(ssc[0].tolist())))
        lines.append('meanIOU: %f\n' % ssc[2])
        lines.append('pixel-accuracy: %f\n' % ssc[3])
        lines.append('')
        lines.append('--*-- Scene Completion --*--\n')
        lines.append('IOU: %f\n' % sc[0])
        lines.append('pixel-accuracy: %f\n' % sc[1])  # 0 和 1 类的IOU
        lines.append('recall: %f\n' % sc[2])

        line = "\n".join(lines)
        print(line)
        return line

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default='last', type=str)
    parser.add_argument('-d', '--devices', default='1', type=str)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('--show_image', '-s', default=False,
                        action='store_true')
    parser.add_argument('--save_path', '-p', default='results')

    args = parser.parse_args()
    all_dev = parse_devices(args.devices)

    network = Network(class_num=config.num_classes, feature=128, bn_momentum=config.bn_momentum,
                norm_layer=nn.BatchNorm3d, eval=True)
    data_setting = {'img_root': config.img_root_folder,
                    'gt_root': config.gt_root_folder,
                    'hha_root':config.hha_root_folder,
                    'mapping_root': config.mapping_root_folder,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source}
    val_pre = ValPre()
    dataset = NYUv2(data_setting, 'val', val_pre)

    with torch.no_grad():
        segmentor = SegEvaluator(dataset, config.num_classes, config.image_mean,
                                 config.image_std, network,
                                 config.eval_scale_array, config.eval_flip,
                                 all_dev, args.verbose, args.save_path,
                                 args.show_image)
        segmentor.run(config.snapshot_dir, args.epochs, config.val_log_file,
                      config.link_val_log_file)
