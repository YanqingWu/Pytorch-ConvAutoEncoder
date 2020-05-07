import cv2
import glob
import numpy as np
from PIL import Image
from os.path import join
from utils.augment import augment
from utils.utils import process_image


class LoadData:
    def __init__(self, root_path='data', img_height=512, img_width=512, mode='train', aug=True):
        self.root_path = root_path
        self.img_height = img_height
        self.img_width = img_width
        self.mode = mode
        self.aug = aug
        self.train_path = join(self.root_path, 'train')
        self.val_path = join(self.root_path, 'val')
        self.test_path = join(self.root_path, 'test')
        # is_valid_file = lambda x: x.endswith('png')

        if mode == 'train':
            self.dataset = glob.glob(self.train_path + '/*.jpg')
        elif mode == 'val':
            self.dataset = glob.glob(self.val_path + '/*.jpg')
        elif mode == 'test':
            self.dataset = glob.glob(self.test_path + '/*.jpg')
        else:
            raise NotImplementedError

    def _augment(self, img):
        if self.mode == 'train':
            if self.aug:
                img = augment(img, (self.img_height, self.img_width))

        return cv2.resize(img, (self.img_height, self.img_width))

    def __getitem__(self, item):
        pil_img = Image.open(self.dataset[item]).convert('RGB')
        img = np.array(pil_img)
        img = self._augment(img)
        img = img.astype(np.uint8)
        img = process_image(img, (self.img_height, self.img_width))
        return img, img

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    data = LoadData()

