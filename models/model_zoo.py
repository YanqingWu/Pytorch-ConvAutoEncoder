from models.unet import UNet
from utils.utils import init_weights
from models.vanilla_ae import AutoEncoder


model_zoo = ['vanilla', 'unet']


def make_model(model_name: str, img_channels=3):
    if model_name.lower() == 'unet':
        model = UNet(num_classes=img_channels)
    elif model_name.lower() == 'vanilla':
        model = AutoEncoder(img_channels=img_channels)
    else:
        raise NotImplemented('model must in %s' % str(model_zoo))
    init_weights(model)
    return model

