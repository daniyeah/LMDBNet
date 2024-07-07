from thop import profile
from torch.utils.data import DataLoader
from torchvision import transforms
from networks.tool.operation import predict
from networks.tool.path import *
from networks.tool.utils import get_net, write_dict_to_excel
import torch
from networks.tool.dataset import RsDataset


def test(args, model_path):
    datas, test_src_t1, test_src_t2, test_label, test_predict, test_lab, test_A, test_B = get_dataset_path(args,
                                                                                                           train=False)
    net, Title = get_net(args)
    save_path = save_path = './' + args['dataset'] + '_result/'
    device = torch.device(f"cuda:{args['gpu']}")
    src_transform = transforms.Compose([transforms.ToTensor(), ])
    label_transform = transforms.Compose([transforms.ToTensor(), ])
    ckps = torch.load(model_path)
    net.load_state_dict(ckps, strict=False)

    dataset_test = RsDataset(test_src_t1, test_src_t2, test_label, test=True, t1_transform=src_transform,
                             t2_transform=src_transform, label_transform=label_transform)
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=4)

    pre_test, rec_test, f1_test, iou_test, kc_test = predict(net, dataloader_test, save_path,device)

    tensor = torch.rand(1, 3, 256, 256).to(device)
    tensor1 = torch.rand(1, 3, 256, 256).to(device)
    flops, params = profile(net, inputs=(tensor1, tensor))
    flops = flops / 1000000000
    params = params / 1000000
    dict1 = {'Net': Title, 'bs': args['bs'], 'lr': args['lr'], 'epo': args['epo'],
             'flops': f'{flops:.3f}', 'params': f'{params:.3f}', 'best_f1': f'{f1_test["f1_1"] * 100:.2f}',
             'best_epo': '', 'date': '', 'time': '', 'dataset': datas, 'illustrate': args['tips'],
             'valortest': 'test'}
    write_dict_to_excel(dict1, './data.xlsx')
    return pre_test['precision_1'], rec_test['recall_1'], f1_test['f1_1'], iou_test['iou_1'], kc_test
