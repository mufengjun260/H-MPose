# --------------------------------------------------------
# DenseFusion 6D Object Pose Estimation by Iterative Dense Fusion
# Licensed under The MIT License [see LICENSE for details]
# Written by Chen
# --------------------------------------------------------
import copy

import cv2
import open3d
from PIL import Image
from PIL import ImageDraw

import _init_paths
import argparse
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
# import torchvision.transforms as transforms
import torchvision.utils as vutils
from datasets.ycb.dataset_test_all import PoseDataset as PoseDataset_ycb
from datasets.linemod.dataset import PoseDataset as PoseDataset_linemod
from lib.network_temporal_cloud_with_emb_T_merge_result import PoseNet, PoseRefineNet
from lib.loss_matrix import Loss
from lib.loss_refiner import Loss_refine
from lib.utils import setup_logger
import torch.multiprocessing
import tensorboardX as tb
import scipy.io as scio
import csv
from lib.knn.__init__ import KNearestNeighbor

# this train script archives with infinite cloud merging and modified PointNet, without motion merging, memory control and weight from input size
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ycb', help='ycb or linemod')
parser.add_argument('--dataset_root', type=str, default='',
                    help='dataset root dir (''YCB_Video_Dataset'' or ''Linemod_preprocessed'')')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
parser.add_argument('--lr', default=0.001, help='learning rate')
parser.add_argument('--lr_rate', default=0.3, help='learning rate decay rate')
parser.add_argument('--w', default=0.015, help='learning rate')
parser.add_argument('--w_rate', default=0.3, help='learning rate decay rate')
parser.add_argument('--decay_margin', default=0.016, help='margin to decay lr & w')
parser.add_argument('--refine_margin', default=0.013, help='margin to start the training of iterative refinement')
parser.add_argument('--noise_trans', default=0.03,
                    help='range of the random noise of translation added to the training data')
parser.add_argument('--iteration', type=int, default=2, help='number of refinement iterations')
parser.add_argument('--nepoch', type=int, default=500, help='max number of epochs to train')
parser.add_argument('--resume_posenet', type=str, default='', help='resume PoseNet model')
parser.add_argument('--resume_refinenet', type=str, default='', help='resume PoseRefineNet model')
parser.add_argument('--start_epoch', type=int, default=1, help='which epoch to start')
parser.add_argument('--output_dir', type=str, default='', help='output dir')
parser.add_argument('--mem_length', type=int, default=5, help='length of history memory')
parser.add_argument('--object_max', type=int, default=21, help='length of classes.txt')

opt = parser.parse_args()
knn = KNearestNeighbor(1)
is_debug = False


def get_target(root, filename, obj):
    meta = scio.loadmat('{0}/{1}-meta.mat'.format(root, filename))
    target_r = None
    target_t = None
    if len(np.where(meta['cls_indexes'].flatten() == int(obj) + 1)[0]) == 1:
        idx = int(np.where(meta['cls_indexes'].flatten() == int(obj) + 1)[0])
        target_r = meta['poses'][:, :, idx][:, 0:3]
        target_t = np.array([meta['poses'][:, :, idx][:, 3:4].flatten()])
    return torch.tensor(target_r).cuda(), torch.tensor(target_t).cuda()


def main():
    class_id = 0
    class_file = open('datasets/ycb/dataset_config/classes.txt')
    output_invible_file = open('invisible_files_occ.csv', 'w', encoding='utf-8')
    result_writer = csv.writer(output_invible_file)
    result_writer.writerow(["frame", "obj_id", "visible_num", "total_num", "invisible_rate", "filename"])
    cld = {}
    while 1:
        class_input = class_file.readline()
        if not class_input:
            break

        input_file = open('{0}/models/{1}/points.xyz'.format(opt.dataset_root, class_input[:-1]))
        cld[class_id] = []
        while 1:
            input_line = input_file.readline()
            if not input_line:
                break
            input_line = input_line[:-1].split(' ')
            cld[class_id].append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
        cld[class_id] = np.array(cld[class_id])
        input_file.close()

        class_id += 1

    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    symmetry_obj_idx = [12, 15, 18, 19, 20]

    # 根据dataset类型设定参数
    if opt.dataset == 'ycb':
        opt.num_objects = 21  # number of object classes in the dataset
        opt.num_points = 1000  # number of points on the input pointcloud
        opt.outf = 'trained_models/ycb/' + opt.output_dir  # folder to save trained models
        opt.test_output = 'experiments/output/ycb/' + opt.output_dir
        if not os.path.exists(opt.test_output): os.makedirs(opt.test_output, exist_ok=True)

        opt.repeat_epoch = 1  # number of repeat times for one epoch training
    elif opt.dataset == 'linemod':
        opt.num_objects = 13
        opt.num_points = 500
        opt.outf = 'trained_models/linemod'
        opt.log_dir = 'experiments/logs/linemod'
        opt.repeat_epoch = 20
    else:
        print('Unknown dataset')
        return

    # 加载数据集
    if opt.dataset == 'ycb':
        dataset = PoseDataset_ycb('train', opt.num_points, False, opt.dataset_root, opt.noise_trans, False,
                                  gaussian=False)
    elif opt.dataset == 'linemod':
        dataset = PoseDataset_linemod('train', opt.num_points, False, opt.dataset_root, opt.noise_trans,
                                      False)
    if opt.dataset == 'ycb':
        test_dataset = PoseDataset_ycb('test', opt.num_points, False, opt.dataset_root, 0.0, False,
                                       gaussian=False)
    elif opt.dataset == 'linemod':
        test_dataset = PoseDataset_linemod('test', opt.num_points, False, opt.dataset_root, 0.0, False)
    testdataloader = torch.utils.data.DataLoader(test_dataset, shuffle=False,
                                                 num_workers=opt.workers)
    # 获取对称图形的数据标号
    opt.sym_list = dataset.get_sym_list()
    opt.num_points_mesh = dataset.get_num_points_mesh()

    print(
        '>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the training set: {0}\nlength of the testing set: {1}\nnumber of sample points on mesh: {2}\nsymmetry object list: {3}'.format(
            len(dataset), len(test_dataset), opt.num_points_mesh, opt.sym_list))

    logger = setup_logger('final_eval_tf_with_seg_square',
                          os.path.join(opt.test_output, 'final_eval_tf_with_seg_square.txt'))

    st_time = time.time()
    isFirstInitLastDatafolder = True
    with torch.no_grad():
        for j, data in enumerate(testdataloader, 0):
            if opt.dataset == 'ycb':
                visible_label, filename, list_idx, list_full_img, list_label_obj = data
            for list_index in range(len(list_idx)):
                idx = list_idx[list_index]
                this_label = list_label_obj[list_index]
                # gt_r, gt_t = get_target(opt.dataset_root, filename[0], idx)
                # if gt_r is None: print('gtr is None')

                visible_num = (visible_label == (idx.item() + 1)).sum().float()
                total_num = (this_label != 0).sum().float()
                visible_rate = visible_num / total_num
                # calc invisible surface rate
                # ["frame", "obj_id", "invisible_rate", "filename"]
                print(filename[0], idx.item(), visible_rate.item())
                result_writer.writerow([filename[0], idx.item(), visible_num, total_num, visible_rate.item(),
                                        '{0}/{1}-color-masked-square.png'.format(opt.dataset_root, filename[0])])
    output_invible_file.close()


def get_gt_cloud(gt_r, gt_t, model_x):
    model_x = torch.tensor(model_x, dtype=torch.float64).cuda()
    target = torch.add(torch.mm(model_x, gt_r.transpose(0, 1)), gt_t)
    return target


if __name__ == '__main__':
    main()
