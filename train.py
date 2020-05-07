from utils.utils import *
from utils.config import parser
from utils.logger import Logger
from utils.trainer import Trainer
from utils.data_loader import LoadData
from models.vanilla_ae import AutoEncoder


def main(args):
    """
    receive many args to train
    """

    """ set logger """
    logger = Logger(args.log_file_path, logger_name=args.logger_name, filename=args.log_file_name)
    logger.log.info(args)
    logger.log.info('log files saved to: %s ' % args.log_file_path)
    logger.log.info('experiment id code: %s' % logger.hash_code)

    """ set seeds """
    init_seeds(args)
    logger.log.info('set seed %s' % args.manual_seed)

    """ load data """
    train_loader, val_loader = get_loaders(args, LoadData)
    logger.log.info('success load data from: %s' % args.data)

    """ choose model """
    # logger.log.info('using model: %s' % args.arc)
    if args.resume:
        logger.log.info('using resume model: %s' % args.resume)
        states = torch.load(args.resume)
        model = states['arc']
        model.load_state_dict(states['state_dict'])
    else:
        logger.log.info('not using resume model')
        model = AutoEncoder()

    """ set cuda """
    use_cuda, multi_gpus = use_gpu_or_multi_gpus(args, logger)
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        if multi_gpus:
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.cuda()

    """ set criterion """
    criterion = nn.MSELoss()
    if use_cuda:
        criterion = criterion.cuda()

    """ set optimizer"""
    if args.linear_scaling:
        args.lr = 0.1 * args.train_batch / 256
    logger.log.info('initial lr: %4f' % args.lr)
    optimizer = get_optimizer(args, model, logger)

    """ low precision training """
    use_low_precision_training = args.low_precision_training
    if use_low_precision_training:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    """ lr scheduler """
    scheduler = get_scheduler(args, optimizer, logger)

    """ train """
    trainer = Trainer(model, criterion, optimizer, logger, scheduler, train_loader, val_loader, use_cuda)
    logger.log.info('start training ...')
    trainer.train(args.epochs)
    logger.log.info('training finished .')


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
