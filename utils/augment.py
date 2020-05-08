from albumentations import *


def augment(img):
    aug = Compose([
        OneOf([RandomGamma(gamma_limit=(80, 120), p=0.5),
               RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2)],
              p=1),
        OneOf([GaussNoise(var_limit=(10.0, 50.0), mean=0, p=0.5),
               ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5),
               MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=False, elementwise=False, p=0.5)],
              p=0.5),
        Flip(p=0.5),
        Transpose(p=0.5),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.5, rotate_limit=45, p=0.5),
        OneOf([MotionBlur(p=0.5),
               MedianBlur(blur_limit=3, p=0.5),
               Blur(blur_limit=3, p=0.5),
               GaussianBlur(blur_limit=3, p=0.5)],
              p=0.5),
        OneOf([OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=1),
               GridDistortion(num_steps=5, distort_limit=0.3, p=1),
               ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5)],
              p=1),
        # color jitter
        OneOf([HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
               CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
               IAASharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),
               IAAEmboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=0.5),
               RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
               ToGray(p=0.5),
               ToSepia(p=0.5),
               InvertImg(p=0.5),
               IAAPerspective(scale=(0.05, 0.1), keep_size=True, p=0.5)],
              p=0.5),
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
    plt.show()

