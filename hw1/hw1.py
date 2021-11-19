"""
姓名：林堃
学号：3019244362
"""

import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import time
from decimal import *


def judge(img1, img2, ratio):
    img1 = img1.astype(np.int32)
    img2 = img2.astype(np.int32)
    with open('img1.txt', 'wt') as f:
        print(img1, file=f)
    with open('img2.txt', 'wt') as f:
        print(img2, file=f)

    diff = np.abs(img1 - img2)
    print(diff)
    count = np.sum(diff > 1)

    assert count == 0, f'ratio={ratio}, Error!'
    print(f'ratio={ratio}, Success!')


def get_gt(img, ratio):
    new_h = int(img.shape[0] * ratio)
    new_w = int(img.shape[1] * ratio)
    gt = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    return gt

def resize(src_img, ratio):
    scrH, scrW, _ = src_img.shape
    dst_shape = [int(scrH * ratio), int(scrW * ratio)]
    dst_img = np.zeros((dst_shape[0], dst_shape[1], 3), np.uint8)
    dst_h, dst_w = dst_shape
    src_h = src_img.shape[0]
    src_w = src_img.shape[1]
    # i：纵坐标y，j：横坐标x
    # 缩放因子，dw,dh
    scale_w = float(src_w / dst_w)
    scale_h = float(src_h / dst_h)

    for i in range(dst_h):
        for j in range(dst_w):
            src_x = float((j + 0.5) * scale_w - 0.5)
            src_y = float((i + 0.5) * scale_h - 0.5)
			# 向下取整，代表靠近源点的左上角的那一点的行列号	
            src_x_int = math.floor(src_x)
            src_y_int = math.floor(src_y)
            # 取出小数部分，用于构造权值
            src_x_float = (src_x - src_x_int)
            src_y_float = (src_y - src_y_int)

            if (src_x_int < 0):
                src_x_float = 0.
                src_x_int = 0
            if (src_x_int >= src_w - 1):
                src_x_float = 1.
                src_x_int = src_w - 2

            if (src_y_int < 0):
                src_y_float = 0.
                src_y_int = 0
            if (src_y_int >= src_h - 1):
                src_y_float = 1.
                src_y_int = src_h - 2

            dst_img[i, j, :] = (1. - src_y_float) * (1. - src_x_float) * src_img[src_y_int, src_x_int, :] + \
                               (1. - src_y_float) * src_x_float * src_img[src_y_int, src_x_int + 1, :] + \
                               src_y_float * (1. - src_x_float) * src_img[src_y_int + 1, src_x_int, :] + \
                               src_y_float * src_x_float * src_img[src_y_int + 1, src_x_int + 1, :]
    return dst_img





def show_images(img1, img2):
    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.show()


if __name__ == '__main__':
    ratios = [0.5, 0.8, 1.2, 1.5]

    # type(img) = ndarray, 一共有三张图片，都可以尝试
    img = cv2.imread('images/img_1.jpeg')

    print(img.shape)

    start_time = time.time()
    for ratio in ratios:
        gt = get_gt(img, ratio)
        resized_img = resize(img, ratio)
        show_images(gt, resized_img)  # added
        judge(gt, resized_img, ratio)
    end_time = time.time()
    total_time = end_time - start_time

    print(f'用时{total_time:.4f}秒')
    print('Pass')
