from main import main
import argparse

def get_params():
    parser = argparse.ArgumentParser(description='RSCD_PyTorch')
    parser.add_argument('--bs', type=int, default=8, metavar='N')
    parser.add_argument('--lr', type=float, default=0.003, metavar='LR')
    parser.add_argument('--epo', type=int, default=200, metavar='EPO')
    parser.add_argument('--gpu', type=int, default=1, metavar='GPU')
    parser.add_argument('--tips', type=str, default='None', metavar='TIPs')
    parser.add_argument('--net', type=str, default='LMDBNet', metavar='Net')
    parser.add_argument('--dataset', type=str, default='LEV', metavar='Dataset')
    parser.add_argument('--deep', type=int, default=1, metavar='Deeper')
    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    params = vars(get_params())
    main(params)
