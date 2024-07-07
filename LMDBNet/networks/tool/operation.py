import os

import numpy as np
import torch
from tqdm import tqdm
from scipy.stats import norm
from networks.tool import metrics
from networks.tool.utils import save_pre_result, save_img
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



def train(net, dataloader_train, criterion_ce, optimizer, scheduler, device):
    print('Training...')
    model = net.train()
    num = 0
    epoch_loss = 0
    cm_total = np.zeros((2, 2))
    for x1, x2, y in tqdm(dataloader_train):
        inputs_t1 = x1.to(device)
        inputs_t2 = x2.to(device)
        labels = y.to(device)
        optimizer.zero_grad()
        pre = model(inputs_t1, inputs_t2)
        loss = criterion_ce(pre, labels)
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
        # pre = torch.max(pre, 1)[1]  # out_ch=2
        cm = metrics.ConfusionMatrix(2, pre[0], labels)
        cm_total += cm
        num += 1

    precision_total, recall_total, f1_total, iou_total, kc_total = metrics.get_score_sum(cm_total)
    scheduler.step(f1_total['f1_1'])

    return epoch_loss, precision_total['precision_1'], recall_total['recall_1'], f1_total['f1_1'], iou_total[
        'iou_1'], kc_total


def deep_train(net, dataloader_train, criterion_ce, optimizer, scheduler, device):
    print('Training...')
    model = net.train()
    num = 0
    epoch_loss = 0
    cm_total = np.zeros((2, 2))
    for x1, x2, y in tqdm(dataloader_train):
        inputs_t1 = x1.to(device)
        inputs_t2 = x2.to(device)
        labels = y.to(device)
        optimizer.zero_grad()
        pre = model(inputs_t1, inputs_t2)
        loss = criterion_ce(pre, labels)  # out_ch=1 DeepLoss
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
        # pre = torch.max(pre, 1)[1]  # out_ch=2
        cm = metrics.ConfusionMatrix(2, pre[0], labels)
        cm_total += cm
        num += 1

    precision_total, recall_total, f1_total, iou_total, kc_total = metrics.get_score_sum(cm_total)
    scheduler.step(f1_total['f1_1'])

    return epoch_loss, precision_total['precision_1'], recall_total['recall_1'], f1_total['f1_1'], iou_total[
        'iou_1'], kc_total


def validate(net, dataloader_val, epoch, device):
    print('Validating...')
    model = net.eval()
    num = 0
    cm_total = np.zeros((2, 2))

    for x1, x2, y in tqdm(dataloader_val):
        inputs_t1 = x1.to(device)
        inputs_t2 = x2.to(device)
        labels = y.to(device)
        pre = model(inputs_t1, inputs_t2)
        # pre = torch.max(pre, 1)[1]  # out_ch=2
        cm = metrics.ConfusionMatrix(2, pre[0], labels)
        cm_total += cm
        num += 1
    precision_total, recall_total, f1_total, iou_total, kc_total = metrics.get_score_sum(cm_total)
    return precision_total['precision_1'], recall_total['recall_1'], f1_total['f1_1'], iou_total['iou_1'], kc_total

def concatenate_tensors(tensor_list, num_per_group):
    group_list = []
    for i in range(0, len(tensor_list), num_per_group):
        group = tensor_list[i:i + num_per_group]
        group_list.append(group)
    return group_list

def predict(net, dataloader_test, save_path, device):
    print('Testing...')
    model = net.eval()
    num = 0
    cm_total = np.zeros((2, 2))
    for x1, x2, y in tqdm(dataloader_test):
        # img_name = dataloader_test.sampler.data_source.file_list[num]
        # print(img_name)
        inputs_t1 = x1.to(device)
        inputs_t2 = x2.to(device)
        labels = y.to(device)

        pre = model(inputs_t1, inputs_t2)
        cm = metrics.ConfusionMatrix(2, pre[0], labels)
        cm_total += cm
        # save_four_colors_results(pre[0],y,save_path,img_name)
        num += 1

    precision_total, recall_total, f1_total, iou_total, kc_total = metrics.get_score_sum(cm_total)
    return precision_total, recall_total, f1_total, iou_total, kc_total


class DeepLoss():
    def __init__(self,deeper):
        self.deeper = deeper
    def __call__(self, pres, label):
        loss = 0
        for i in range(self.deeper):
            loss += F.binary_cross_entropy(pres[i], label)
        return loss



def save_four_colors_results(pred, label,save_path,img_name):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    pr = pred[0, 0].detach().cpu().numpy()
    gt = label[0, 0].cpu().numpy()


    pr[pr >= 0.5] = 1
    pr[pr < 0.5] = 0
    gt[gt >= 0.5] = 1
    gt[gt < 0.5] = 0


    index_tp = np.where(np.logical_and(pr == 1, gt == 1))
    index_fp = np.where(np.logical_and(pr == 1, gt == 0))

    index_tn = np.where(np.logical_and(pr == 0, gt == 0))
    index_fn = np.where(np.logical_and(pr == 0, gt == 1))

    map = np.zeros([gt.shape[0], gt.shape[1], 3])

    map[index_tp] = [255, 255, 255]  # white
    map[index_fp] = [255, 0, 0]  # red
    map[index_tn] = [0, 0, 0]  # black
    map[index_fn] = [0, 255, 0]  # Cyan
    change_map = Image.fromarray(np.array(map, dtype=np.uint8))
    change_map.save(save_path + img_name)
