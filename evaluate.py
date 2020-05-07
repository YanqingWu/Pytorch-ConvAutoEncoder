import glob
import torch
import argparse
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from predict import predict
from utils.utils import PathLoader
from torch.utils.data import DataLoader


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def evaluate(model, imgs):
    criterion = nn.MSELoss(reduction='none')
    preds, results = predict(model, imgs)
    reconstruct_error = criterion(imgs.to(device), results)
    reconstruct_error = reconstruct_error.mean(dim=(1, 2, 3)).tolist()
    return reconstruct_error


def evaluate_path(path, model, batch_size=16):
    res = []
    loader = PathLoader(path)
    batch_loader = DataLoader(loader, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)
    indexs = [file.split('/')[-1] for file in loader.images]
    for imgs in tqdm(batch_loader):
        reconstruct_error = evaluate(model, imgs)
        res.extend(reconstruct_error)
    df = pd.DataFrame(res, index=indexs, columns=['reconstruct error'])
    df.to_csv(path + '/a0_results.csv')


def evaluate_root(root, model, batch_size=16):
    for path in glob.glob(root + '/*'):
        evaluate_path(path, model, batch_size=batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('evaluate')
    parser.add_argument('--load-model', default='', type=str,
                        help='trained model path')
    parser.add_argument('--batch-size', default=16, type=int,
                        help='batch size')
    parser.add_argument('--data-root', type=str, default='',
                        help='root path contains folders')
    parser.add_argument('--data-path', type=str, default='',
                        help='path contains images')

    args = parser.parse_args()
    state = torch.load(args.load_model)
    model = state['arc'].to(device)
    model.load_state_dict(state['state_dict'])
    print('loaded model: %s' % args.load_model)
    model.eval()

    if args.data_root:
        evaluate_root(args.data_root, model, batch_size=args.batch_size)
    else:
        evaluate_path(args.data_path, model, batch_size=args.batch_size)

