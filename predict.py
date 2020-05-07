import cv2
import torch
import argparse
import numpy as np
from utils.utils import process_image, inverse_process


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def predict(model, img):
    model.eval()
    with torch.no_grad():
        h, w, c = img.shape
        img = process_image(img, (512, 512))
        img = img.unsqueeze(0)
        res = model(img.to(device))
        res = inverse_process(res[0])
        res = cv2.resize(res, (w, h))
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser('predict')
    parser.add_argument('--load-model', default='', type=str,
                        help='trained model path')
    parser.add_argument('--batch-size', default=16, type=int,
                        help='batch size')
    # parser.add_argument('--data-root', type=str, default='',
    #                     help='root path contains folders')
    # parser.add_argument('--data-path', type=str, default='',
    #                     help='path contains images')

    args = parser.parse_args()
    import matplotlib.pyplot as plt
    state = torch.load(args.load_model)
    model = state['arc'].to(device)
    model.load_state_dict(state['state_dict'])
    print('loaded model: %s' % args.load_model)
    model.eval()
    img = cv2.imread('data/train/1_20-01-06_14_18_46_b.jpg')
    pred = predict(model, img)
    plt.imshow(img)
    plt.show()
    plt.imshow(pred)
    plt.show()

