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
    reconstruct_error = nn.MSELoss()(img.to(device), results)
    return reconstruct_error, img_copy, pred


if __name__ == '__main__':
    import torch.nn as nn
    from PIL import Image
    import matplotlib.pyplot as plt
    state = torch.load('trained_models/73b6b4bd_AutoEncoder_loss_0.0337_model_best.pth')
    model = state['arc'].to(device)
    model.load_state_dict(state['state_dict'])
    model.eval()
    # img = Image.open('data/test/left_01-16_14_58_41_059.jpg').convert('RGB')
    img = Image.open('data/val/67_20-01-07_03_38_42_a.jpg').convert('RGB')
    img = np.array(img)
    reconstruct_error, img_copy, pred = evaluate(model, img)

    print('reconstruct error: %.4f' % reconstruct_error)
    plt.imshow(pred)
    plt.show()


