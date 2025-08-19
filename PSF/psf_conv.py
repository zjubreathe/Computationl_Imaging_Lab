from skimage import color
from skimage.io import imread, imsave
from scipy.signal import convolve2d
from PIL import Image
import scipy.io as scio
from scipy import signal
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import os
from numpy import fft
import math
import cv2
import time


# 改进的平滑卷积方式
def Convolutional_3(raw_img, PSF):
    global fov_num_x
    global fov_num_y
    global psf_shape
    H, W = raw_img.shape
    h, w = psf_shape, psf_shape
    aa = int(h/2)
    img = np.zeros((H-h+1,W-w+1)).astype(np.uint8)
    block_shape = int((H-h+1) / fov_num_y)
    # print(H, W, h, w, img.shape, block_shape, fov_num_x * fov_num_y)
    psf_linear = np.zeros((H-2*h+1, W-2*w+1, h, w))
    for i in range(H-2*h+1):
        for j in range (W-2*w+1):
            # print(i,j)
            a_1 = i // psf_shape
            a_2 = i % psf_shape
            b_1 = j // psf_shape
            b_2 = j % psf_shape
            PSF_num = a_1 * fov_num_x + b_1
            # 对每一点的PSF进行加权，线性插值
            psf_linear[i, j, :, :] = ((psf_shape - b_2) / psf_shape * PSF[PSF_num, :, :] + \
                                      b_2 / psf_shape * PSF[PSF_num + 1, :, :]) * (psf_shape - a_2) / psf_shape + \
                                     ((psf_shape - b_2) / psf_shape * PSF[PSF_num + fov_num_x, :, :] + \
                                      b_2 / psf_shape * PSF[PSF_num + fov_num_x + 1, :, :]) * a_2 / psf_shape
    psf_linear = np.pad(psf_linear, ((aa, aa), (aa, aa), (0, 0), (0, 0)), 'edge')
    # print(psf_linear.shape)
    # print('xxxxxxxx')


    for i in range(H-h+1):
        for j in range (W-w+1):
            # block = np.zeros((h,w)).astype(np.uint8)
            raw_block = raw_img[i: i + psf_shape, j: j + psf_shape]
            PSF_ave = psf_linear[i, j, :, :]
            aa = convolve2d(raw_block, PSF_ave, mode='valid')
            img[i,j] = aa
    return img



# R=scio.loadmat("VIS_PSF_real.mat")
R=scio.loadmat("psf.mat")
PSF = np.array(R['normalized_VIS_PSF'])
a1,psf_shape,a2 = PSF.shape
# print(psf_shape)
# psf_shape = 64
fov_num_x = 10
fov_num_y = 8
aa = int(psf_shape/2)
bb = int(psf_shape/2-1)
# file_root = r"C:\Users\Admin\Downloads\FLIR_ADAS_v2\long\test\sharp/"
# save_out = r"C:\Users\Admin\Downloads\FLIR_ADAS_v2\long\test\blur/"
file_root = r"sharp2/"
save_out = r"blur2/"

file_list = os.listdir(file_root)

for img_name in tqdm(file_list):
    # print(psf_shape)
    img_path = file_root + img_name
    out_path = save_out + img_name
    if img_name.endswith('.jpg'):
        color_img = cv2.imread(img_path)
        gray_img = cv2.cvtColor(color_img,cv2.COLOR_BGR2GRAY)
        # print(gray_img.shape)
        raw_img = np.array(gray_img)
        raw_img = np.pad(raw_img, ((aa,bb),(aa,bb)), 'edge')
        # print(raw_img.shape)
        raw_img1 = Convolutional_3(raw_img, PSF)
        out_name = img_name.split('.')[0]
        save_path1 = save_out + out_name + '.jpg'
        imsave(save_path1, raw_img1)

