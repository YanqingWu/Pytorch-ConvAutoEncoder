from albumentations import *


def augment(img):
    aug = Compose([
        RandomRotate90(),
        OneOf([RandomGamma(), RandomBrightnessContrast()], p=1),
        OneOf([GaussNoise(), ISONoise(), MultiplicativeNoise()], p=0.5),
        Flip(),
        Transpose(),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.5, rotate_limit=45, p=0.5),
        OneOf([MotionBlur(p=0.5), MedianBlur(blur_limit=3, p=0.5), Blur(blur_limit=3, p=0.5),
               GaussianBlur(blur_limit=3, p=0.5)], p=0.5),
        # OneOf([OpticalDistortion(p=0.5), GridDistortion(p=0.5)], p=0.5),
        # color jitter
        OneOf([HueSaturationValue(), CLAHE(clip_limit=2), IAASharpen(), IAAEmboss(),
               RandomBrightnessContrast(), RGBShift(), ToGray(), ToSepia(), InvertImg(),
               IAAPerspective(), ElasticTransform()], p=0.5),
    ], p=1)
    return aug(**{'image': img})['image']


if __name__ == '__main__':
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt
    img = Image.open('data/val/67_20-01-07_03_38_42_a.jpg').convert('RGB')
    img = np.array(img)
    img = augment(img)
    plt.imshow(img)

