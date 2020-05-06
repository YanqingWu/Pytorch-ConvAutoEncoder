import os
import sys
import time
import glob
import torch
import logging
import hashlib
from tensorboardX import SummaryWriter


class Logger:
    def __init__(self, path='logs', logger_name='train', filename='train.txt'):
        if not os.path.exists(path):
            os.mkdir(path)
        self.log = logging.getLogger(logger_name)
        self.log.setLevel(level=logging.INFO)
        md = hashlib.md5()
        md.update(str(time.time()).encode('utf-8'))
        self.hash_code = md.hexdigest()[:8]
        self.log_path = path + '/' + self.hash_code
        if not os.path.exists(self.log_path):
            os.mkdir(self.log_path)
        # writer to file
        file_handler = logging.FileHandler(self.log_path + '/' + filename)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        # write to console
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level=logging.INFO)
        self.log.addHandler(stream_handler)
        self.log.addHandler(file_handler)
        self.writter = SummaryWriter(log_dir=self.log_path + '/runs')

    def _remove_files(self, file_list):
        for f in file_list:
            os.remove(f)

    def save_checkpoint(self, state, is_best=False, save_path='trained_models', prefix=''):
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        if is_best:
            self._remove_files(glob.glob(save_path + '/%s*%s' % (self.hash_code, 'model_best.pth')))
            torch.save(state, save_path + '/%s_%s_model_best.pth' % (self.hash_code, prefix))
        else:
            self._remove_files(glob.glob(save_path + '/%s*%s' % (self.hash_code, 'model_latest.pth')))
            torch.save(state, save_path + '/%s_%s_model_latest.pth' % (self.hash_code, prefix))
