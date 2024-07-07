import os
import torch
import torch.nn as nn
from thop import profile
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from networks.tool.dataset import RsDataset
from networks.tool.operation import *
from networks.tool.path import get_dataset_path
from networks.tool.utils import RandomVerticalFlip, RandomHorizontalFlip, ToTensor, RandomFixRotate, get_net, \
    get_realated_imformation, get_logger, write_dict_to_excel
from networks.tool.predict import test


def main(args):
    """===prepare for the training==="""
    net, TITLE = get_net(args)
    device = torch.device(f"cuda:{args['gpu']}")
    today, start_time, start_epoch, best_f1, best_epoch = get_realated_imformation()
    datas, train_src_t1, train_src_t2, train_label, val_src_t1, val_src_t2, val_label = get_dataset_path(args,
                                                                                                         train=True)

    criterion_ce = DeepLoss(args['deep'])
    optimizer = optim.AdamW(net.parameters(), args['lr'], weight_decay=0.0005, )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.3, patience=10,
                                                           threshold=0.0015, cooldown=5, min_lr=2e-7, )
    #   数据集的导入
    src_transform = transforms.Compose([RandomHorizontalFlip(), RandomVerticalFlip(), RandomFixRotate(), ToTensor()])
    label_transform = transforms.Compose([transforms.ToTensor()])

    dataset_train = RsDataset(train_src_t1, train_src_t2, train_label, test=False, t1_transform=src_transform,
                              t2_transform=src_transform, label_transform=label_transform)
    dataset_val = RsDataset(val_src_t1, val_src_t2, val_label, test=True, t1_transform=label_transform,
                            t2_transform=label_transform, label_transform=label_transform)
    dataloader_train = DataLoader(dataset_train, batch_size=args['bs'], shuffle=True, num_workers=4)
    dataloader_val = DataLoader(dataset_val, batch_size=4, shuffle=False, num_workers=4)
    num_dataset = len(dataloader_train.dataset)
    total_step = (num_dataset - 1) // dataloader_train.batch_size + 1

    # 日志文件保存位置
    if not os.path.exists('logs/' + str(today)):
        os.makedirs('logs/' + str(today))
    ckp_savepath = 'ckps/' + str(today) + '/' + TITLE
    logger = get_logger(f'logs/{str(today)}/{TITLE}_{start_time}_{datas}.log')
    logger.info(
        f"Net: {TITLE}\t Batch Size: {args['bs']}\t\t Learning Rate:{args['lr']}\t epoch:{args['epo']}\t\t "
        f"save_path:{ckp_savepath}\t GPU:{args['gpu']}\n提示信息：{args['tips']}\nDataset:{datas}\ndeep:{args['deep']}")
    if not os.path.exists(ckp_savepath):
        os.makedirs(ckp_savepath)

    # print('==> 参数及计算量评估...')
    # tensor = torch.rand(1, 3, 256, 256).to(device)
    # tensor1 = torch.rand(1, 3, 256, 256).to(device)
    # flops, params = profile(net, inputs=(tensor1, tensor))
    # flops = flops / 1000000000
    # params = params / 1000000
    # logger.info(f'flops:{flops:.4f}\tparams:{params:.4f}')
    ckp_name = ''
    #   开始训练
    for epoch in range(start_epoch, args['epo']):
        print(f"Epoch {epoch + 1}/{args['epo']}\n" + '=' * 10)
        epoch += 1
        loss_train, pre_train, recall_train, f1_train, iou_train, kc_train = train(net, dataloader_train, criterion_ce,
                                                                                   optimizer, scheduler, device)
        pre_val, recall_val, f1_val, iou_val, kc_val = validate(net, dataloader_val, epoch, device)

        # 训练结果记录
        if f1_val > best_f1:
            best_f1 = f1_val
            best_epoch = epoch
            ckp_name = f"{start_time}_batch={args['bs']}_lr={args['lr']}_epoch{epoch}_{datas}.pth"
            torch.save(net.state_dict(), os.path.join(ckp_savepath, ckp_name), _use_new_zipfile_serialization=False)
        logger.info(
            f"Epoch:[{epoch}/{args['epo']}]\t loss_train={loss_train / total_step:.5f}\t"
            f' train_Pre={pre_train:.4f}\t train_Rec={recall_train:.4f}\t train_F1={f1_train:.4f}\t '
            f"train_IoU={iou_train:.4f}\t train_KC={kc_train:.4f}\tlr={scheduler.optimizer.param_groups[0]['lr']}")
        logger.info(
            f"Epoch:[{epoch}/{args['epo']}]\t\t\t val_Pre={pre_val:.4f}\t val_Rec={recall_val:.4f}\t val_F1={f1_val:.4f}\t"
            f' IoU={iou_val:.4f}\t KC={kc_val:.4f}\tbest_F1:[{best_f1:.4f}/{best_epoch}]\t')
    logger.info(
        f'Done\n====start test====\nDataset: {datas}\nNet: {TITLE}\nEpoch: {best_epoch}\nload_dir: {os.path.join(ckp_savepath, ckp_name)}')
    pre_test, rec_test, f1_test, iou_test, kc_test = test(args, os.path.join(ckp_savepath, ckp_name))
    logger.info(
        f'test_Pre={pre_test:.4f}\t\t test_Rec:{rec_test:.4f}\t\t test_F1={f1_test:.4f}\t\t IoU={iou_test:.4f}\t\t KC={kc_test:.4f}\n====Test Done====')
