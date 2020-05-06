from albumentations import *


def augment(img, input_shape):
    assert len(input_shape) == 2
    aug = Compose([  # CenterCrop(input_shape[0], input_shape[1], p=1),
        # RandomRotate90(),
        # OneOf([RandomGamma(), RandomBrightnessContrast()], p=1),
        # OneOf([GaussNoise(), ISONoise(), MultiplicativeNoise()], p=0.5),
        # base
        # Resize(255, 255),
        # OneOf([IAACropAndPad(p=0.5),
        #        CoarseDropout(max_height=int(input_shape[0] / 8), max_width=int(input_shape[1]/8))], p=0.5),
        Flip(),
        Transpose(),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.5, rotate_limit=45, p=0.5)
    ], p=1)
    return aug(**{'image': img})['image']

