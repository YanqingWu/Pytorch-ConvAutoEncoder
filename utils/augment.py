from albumentations import *


def augment(img):
    aug = Compose([
        # """ flip or transpose """
        OneOf([HorizontalFlip(p=1),
               VerticalFlip(p=1),
               Transpose(p=1)],
              p=0.5),

        # """ bright or contrast """
        OneOf([RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0., p=1),
               RandomBrightnessContrast(brightness_limit=0., contrast_limit=0.3, p=1)],
              p=1),

        # """ noise """
        OneOf([GaussNoise(var_limit=(10.0, 50.0), mean=0, p=1),
               ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1),
               MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=False, elementwise=False, p=1)],
              p=1),

        # """ scale """
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.5, rotate_limit=1, p=0.5),

        # """ blur """
        OneOf([MotionBlur(p=1),
               MedianBlur(blur_limit=3, p=1),
               Blur(blur_limit=3, p=1),
               GaussianBlur(blur_limit=3, p=1)],
              p=0.5),

        # """ shape distortion """
        OneOf([OpticalDistortion(distort_limit=0.03, shift_limit=0.05, p=1),
               GridDistortion(num_steps=1, distort_limit=0.1, p=1)],
              p=1),

        # """ color jitter """
        OneOf([HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1),
               RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1),
               ToGray(p=1),
               ToSepia(p=1)],
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
