import cv2
import torch
import numpy as np
from utils.utils import process_image, inverse_process


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def predict(model, img: torch.Tensor):
    model = model.to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        results = model(img.to(device))
        for res in results:
            res = inverse_process(res)
            preds.append(res)
    preds = np.array(preds)
    return preds, results


def evaluate(model, img, img_size=(512, 512)):
    h, w, c = img.shape
    img_copy = img.copy()
    img = process_image(img, img_size).unsqueeze(0)
    preds, results = predict(model, img)
    pred = cv2.resize(preds[0], (w, h))
    reconstruct_error = nn.MSELoss(reduction='none')(img.to(device), results)
    return reconstruct_error, img_copy, pred


def plot(file, color):
    img = Image.open(file).convert('RGB')
    img = np.array(img)
    reconstruct_error, img_copy, pred = evaluate(model, img)

    # print('reconstruct error: %.4f' % reconstruct_error)
    # plt.imshow(pred)
    # plt.show()
    reconstruct_error = reconstruct_error.cpu().numpy().ravel()
    # p1, p2 = np.percentile(reconstruct_error, [95, 99])
    # reconstruct_error[reconstruct_error < p1] = 0
    # reconstruct_error[reconstruct_error > p2] = 0
    # reconstruct_error = reconstruct_error[reconstruct_error.nonzero()]
    # reconstruct_error = np.log(reconstruct_error)
    reconstruct_error.sort()
    reconstruct_error = reconstruct_error[-10000:]
    plt.hist(reconstruct_error, bins=1000, color=color, alpha=0.3)
    # counts, bins = np.histogram(reconstruct_error, bins=100)


if __name__ == '__main__':
    import os
    import torch.nn as nn
    from PIL import Image
    import matplotlib.pyplot as plt
    state = torch.load('trained_models/fd334441_UNet_loss_0.0010_model_best.pth')
    model = state['arc'].to(device)
    model.load_state_dict(state['state_dict'])
    model.eval()
    # img = Image.open('data/test/ng.jpg').convert('RGB')
    plt.subplot(121)
    for i in range(1, 10):
        file = 'data/test/ok/%s.jpg' % i
        if not os.path.exists(file):
            break
        plot(file, 'g')
    plt.subplot(122)
    for i in range(1, 10):
        file = 'data/test/ng/%s.jpg' % i
        if not os.path.exists(file):
            break
        plot(file, 'r')
