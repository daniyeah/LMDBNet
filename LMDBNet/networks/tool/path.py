import sys


def get_dataset_path(args, train=True):
    datas = None
    if args['dataset'] == 'LEV':
        datas = 'LEV-png'
    elif args['dataset'] == 'CDD':
        datas = 'CDD'
    elif datas is None:
        print('Can not find Dataset')
        sys.exit()
    dataset = '../dataset/' + datas
    train_root = dataset + '/train'
    train_src_t1 = train_root + '/A'
    train_src_t2 = train_root + '/B'
    train_label = train_root + '/OUT'

    val_root = dataset + '/val'
    val_src_t1 = val_root + '/A'
    val_src_t2 = val_root + '/B'
    val_label = val_root + '/OUT'

    test_root = dataset + '/test'
    test_src_t1 = test_root + '/A'
    test_src_t2 = test_root + '/B'
    test_label = test_root + '/OUT'

    if train:
        print("get training dataset_dir")
        return datas, train_src_t1, train_src_t2, train_label, val_src_t1, val_src_t2, val_label
    else:
        print("get testing dataset_dir")
        return datas, test_src_t1, test_src_t2, test_label
