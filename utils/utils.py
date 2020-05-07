import cv2
import glob
import math
import torch
import random
import numpy as np
import torch.nn as nn
from PIL import Image
import torch.optim as optim
from torch.utils.data import DataLoader


def process_image(img, img_size=()):
    if img_size:
        img = np.uint8(img)
        img = cv2.resize(img, img_size)
    img = img / 255
    img -= np.array([0.485, 0.456, 0.406])
    img /= np.array([0.229, 0.224, 0.225])
    img = torch.from_numpy(img)
    img = img.permute((2, 0, 1)).float()
    return img


def inverse_process(img):
    img = img.permute(1, 2, 0)
    img = img.cpu().numpy()
    img *= np.array([0.229, 0.224, 0.225])
    img += np.array([0.485, 0.456, 0.406])
    img *= 255
    img = np.uint8(img)
    return img


def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


def init_seeds(args):
    if args.manual_seed is None:
        args.manual_seed = random.randint(1, 10000)
    random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    np.random.seed(args.manual_seed)


def use_gpu_or_multi_gpus(args, logger):
    have_cuda = torch.cuda.is_available()
    use_cuda = args.use_gpu and have_cuda
    logger.log.info('using cuda: %s' % use_cuda)
    if have_cuda and not use_cuda:
        logger.log.info('\nWARNING: found gpu but not use, you can switch it on by: -ug or --use-gpu\n')

    multi_gpus = False
    if use_cuda:
        if args.multi_gpus:
            gpus = torch.cuda.device_count()
            multi_gpus = gpus > 1
            if multi_gpus:
                logger.log.info('using multi gpus, found %d gpus.' % gpus)
    return use_cuda, multi_gpus


def get_loaders(args, loader):
    train_data = loader(mode='train', root_path=args.data, img_width=args.img_width,
                        img_height=args.img_height, aug=args.augment)
    val_data = loader(mode='val', root_path=args.data, img_width=args.img_width,
                      img_height=args.img_height, aug=args.augment)
    train_loader = DataLoader(train_data, batch_size=args.train_batch, shuffle=True,
                              drop_last=True, num_workers=args.workers)
    val_loader = DataLoader(val_data, batch_size=args.val_batch, shuffle=True, drop_last=False,
                            num_workers=args.workers)
    return train_loader, val_loader


def no_bias_weight_decay(args, model):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    parameters = [{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                   'weight_decay': args.weight_decay},
                  {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                   'weight_decay': 0.0}]
    return parameters


def get_scheduler(args, optimizer, logger):
    if args.warmup:
        logger.log.info('using CosineAnnealingWarmRestarts scheduler, warmup epochs: %d' % args.warmup_epochs)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.warmup_epochs, 1, 1e-5)
    elif args.cosine:
        logger.log.info('using CosineAnnealingLR lr scheduler')
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    else:
        logger.log.info('using ExponentialLR lr decay scheduler')
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    return scheduler


def get_optimizer(args, model, logger):
    if args.no_bias_decay:
        logger.log.info('using no bias weight decay')
        parameters = no_bias_weight_decay(args, model)
        optimizer = optim.SGD(parameters, lr=args.lr, momentum=args.momentum)
    else:
        logger.log.info('using bias weight decay')
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    return optimizer


class PathLoader(torch.utils.data.Dataset):
    def __init__(self, path, img_size=(512, 512)):
        self.images = glob.glob(path + '/*.jpg')
        self.img_size = img_size

    def __getitem__(self, item):
        img = Image.open(self.images[item]).convert('RGB')
        img = process_image(img, self.img_size)
        return img

    def __len__(self):
        return len(self.images)

