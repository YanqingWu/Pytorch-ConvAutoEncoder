import argparse


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

""" load data args"""
parser.add_argument('-d', '--data', default='data', type=str,
                    help='data root path, have {train, val} under root, '
                         'every class is a single folder under {train, val}.')

parser.add_argument('-j', '--workers', default=4, type=int,
                    help='number of data loading workers, windows need to change to 0.')

parser.add_argument('-ih', '--img-height', default=512, type=int,
                    help='image height')

parser.add_argument('-iw', '--img-width', default=512, type=int,
                    help='image width')

parser.add_argument('-aug', '--augment', action='store_false',
                    help='switch off image augment')
""" lr args """
parser.add_argument('-lr', '--learning-rate', default=0.001, type=float, dest='lr',
                    help='initial learning rate')

parser.add_argument('-wp', '--warmup', action='store_true',
                    help='learning rate warmup')

parser.add_argument('-we', '--warmup-epochs', default=50, type=int,
                    help='learning rate warmup epochs, only if warmup is true')

parser.add_argument('-ls', '--linear-scaling', action='store_true',
                    help='linear scaling learning rate')

parser.add_argument('-cos', '--cosine', action='store_false',
                    help='cosine learning rate decay')

""" low precision training  """
parser.add_argument('-lpt', '--low-precision-training', action='store_true',
                    help='low precision training')

""" model args """
parser.add_argument('-a', '--arc', default='unet',
                    help='model architecture')

""" loss function """
parser.add_argument('--loss', type=str, default='mse',
                    help='loss function accept [MSE, L1, SmoothL1, LogCosh, XTanh, XSigmoid, Algebraic]')

""" train args """
parser.add_argument('-ep', '--epochs', default=200, type=int,
                    help='number of total epochs to run')

parser.add_argument('-tb', '--train-batch', default=6, type=int,
                    help='train batch size')

""" optimizer """
parser.add_argument('-m', '-momentum', default=0.9, type=float, dest='momentum',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay for all parameters, if No bias decay is set False.')
parser.add_argument('-nb', '--no-bias-decay', action='store_false',
                    help='no bias decay')

""" resume """
parser.add_argument('-r', '--resume', type=str,
                    help='resume model')

""" val args """
parser.add_argument('-vb', '--val-batch', default=12, type=int,
                    help='val batch size')

""" seed args """
parser.add_argument('-seed', '--manual-seed', type=int,
                    help='manual seed')

""" gpu args """
parser.add_argument('-ug', '--use-gpu', action='store_false',
                    help='use gpu training')

parser.add_argument('-sgi', '--single-gpu-id', type=int, default=0,
                    help='gpu id for training')

parser.add_argument('-mg', '--multi-gpus', action='store_true',
                    help='multi gpu training')

""" logs """
parser.add_argument('-log_file', '--log_file_path', default='logs',
                    help='where to save logs')
parser.add_argument('-ln', '--logger-name', default='TrainingLog',
                    help='logger name')
parser.add_argument('-lfn', '--log-file-name', default='train.txt',
                    help='log file name')

