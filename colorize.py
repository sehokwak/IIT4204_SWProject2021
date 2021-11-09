import os
import cv2
import numpy as np


LABEL_COLOR = [(0, 0, 0)
               # 0=background
               , (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 0, 128), (192, 128, 128)
               # 1=car, 2=cat, 3=chair, 4=dog, 5=person
               , (0, 128, 128), (64, 128, 128), (0, 192, 0), (128, 192, 0), (0, 64, 128)]
               # 6=bus, 7=motorbike 8=sofa, 9=train, 10=tv/monitor


def convert_color(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    new_image = np.zeros((image.shape[0], image.shape[1], 3))
    for i in range(10):
        new_image[np.where(image == i)] = LABEL_COLOR[i]
    return new_image[:, :, ::-1]


if __name__ == '__main__':

    # Image List
    image_path = 'datalist/PascalVOC/####.txt'
    with open(image_path, 'r') as f:
        data = f.readlines()
    image_list = [name.strip() for name in data]
    final = []
    for img in image_list[4:8]:
        # Path for original image
        origin_path = 'dataset/image_10/'
        # Path for your predicted image
        pred_path = 'log/session/pred_images/'
        # Path for Colorized GT image (if you don't have this folder in your dataset, plz contact TAs)
        gt_path = 'dataset/mask_10'

        origin = os.path.join(origin_path, img+'.jpg')
        origin = cv2.imread(origin, cv2.IMREAD_COLOR)
        origin = cv2.resize(origin, (300, 300))

        pred = os.path.join(pred_path, img+'.png')
        new_pred = convert_color(pred)
        new_pred = cv2.resize(new_pred, (300, 300))

        gt = os.path.join(gt_path, img+'.png')
        gt = cv2.imread(gt, cv2.IMREAD_COLOR)
        gt = cv2.resize(gt, (300, 300))

        bar_h = np.ones((gt.shape[0], 4, 3)) * 255

        final.append(np.concatenate((origin, bar_h, new_pred, bar_h, gt), axis=1))
    final = np.concatenate(final, axis=0)
    cv2.imwrite(os.path.join('log/session/results', 'result2.png'), final)