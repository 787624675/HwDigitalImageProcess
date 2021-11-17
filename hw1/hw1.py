"""
姓名：林堃
学号：3019244362
"""

import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import time


def judge(img1, img2, ratio):
    img1 = img1.astype(np.int32)
    img2 = img2.astype(np.int32)

    diff = np.abs(img1 - img2)
    count = np.sum(diff > 1)

    assert count == 0, f'ratio={ratio}, Error!'
    print(f'ratio={ratio}, Success!')


def get_gt(img, ratio):
    new_h = int(img.shape[0] * ratio)
    new_w = int(img.shape[1] * ratio)
    gt = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    return gt


# to do
def resize(img, ratio):
    """
    禁止使用cv2、torchvision等视觉库
    type img: ndarray(uint8)
    type ratio: float
    rtype: ndarray(uint8)
    """
    pass


def show_images(img1, img2):
    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.show()


if __name__ == '__main__':
    ratios = [0.5, 0.8, 1.2, 1.5]

    img = cv2.imread('images/img_1.jpeg')   # type(img) = ndarray, 一共有三张图片，都可以尝试

    start_time = time.time()
    for ratio in ratios:
        gt = get_gt(img, ratio)
        resized_img = resize(img, ratio)

        judge(gt, resized_img, ratio)
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f'用时{total_time:.4f}秒')
    print('Pass')
