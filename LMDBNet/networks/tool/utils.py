import datetime
import logging
import random
import sys
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from openpyxl import load_workbook
from networks import LMDBNet

random_rate = 0.5
def get_net(args):
    net = None
    print(args['net'])
    if args['net'] =="LMDBNet":
        net = LMDBNet.LMDBNet()
    elif net is None:
        print("Can not find the Network")
        sys.exit()
    TITLE = net.__class__.__name__
    print('CUDA: ', torch.cuda.is_available())
    device = torch.device(f"cuda:{args['gpu']}")
    net = net.to(device)
    return net, TITLE


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    logger.addHandler(sh)

    return logger


def to8bits(img):
    result = np.ones([img.shape[0], img.shape[1]], dtype='int')
    result[img == 0] = 0
    result[img == 1] = 255
    return result


def save_pre_result(pre, flag, num, save_path):
    pre[pre >= 0.5] = 255
    pre[pre < 0.5] = 0
    outputs = torch.squeeze(pre).cpu().detach().numpy()
    outputs = Image.fromarray(np.uint8(outputs))
    outputs.save(save_path + '/%s_%d.png' % (flag, num))


def save_img(img, flag, num, save_path):
    outputs = torch.squeeze(img).cpu()
    outputs = T.ToPILImage()(outputs)
    outputs.save(save_path + '/%s_%d.png' % (flag, num))


def write_dict_to_excel(data_dict, filename):
    workbook = load_workbook(filename)
    worksheet = workbook.active
    values = list(data_dict.values())
    worksheet.append(values)
    workbook.save(filename)


class RandomHorizontalFlip(object):
    def __call__(self, img):
        if random.random() < random_rate:
            img['img_t1'] = img['img_t1'].transpose(Image.FLIP_LEFT_RIGHT)
            img['img_t2'] = img['img_t2'].transpose(Image.FLIP_LEFT_RIGHT)
            img['img_y'] = img['img_y'].transpose(Image.FLIP_LEFT_RIGHT)
        return img


class RandomVerticalFlip(object):
    def __call__(self, img):
        if random.random() < random_rate:
            img['img_t1'] = img['img_t1'].transpose(Image.FLIP_TOP_BOTTOM)
            img['img_t2'] = img['img_t2'].transpose(Image.FLIP_TOP_BOTTOM)
            img['img_y'] = img['img_y'].transpose(Image.FLIP_TOP_BOTTOM)
        return img


class RandomFixRotate(object):
    def __init__(self):
        self.degree = [Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]

    def __call__(self, img):
        if random.random() < random_rate:
            rotate_degree = random.choice(self.degree)
            img['img_t1'] = img['img_t1'].transpose(rotate_degree)
            img['img_t2'] = img['img_t2'].transpose(rotate_degree)
            img['img_y'] = img['img_y'].transpose(rotate_degree)

        return img


class ToTensor():
    def __call__(self, img):
        img['img_t1'] = transforms(img['img_t1'])
        img['img_t2'] = transforms(img['img_t2'])
        img['img_y'] = transforms(img['img_y'])
        return img


transforms = T.Compose([T.ToTensor()])


def get_realated_imformation():
    today = datetime.date.today()
    start_time = datetime.datetime.now().strftime("%H:%M")
    start_epoch = 0
    best_f1 = 0
    best_epoch = 0
    return today, start_time, start_epoch, best_f1, best_epoch


