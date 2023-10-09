from cv2 import CV_32F, magnitude
import numpy as np
import cv2
import matplotlib.pyplot as plt

#pip install opencv-python

# 影像梯度運算子
def sobel_gradient(img):
    sobel_x = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    sobel_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    grad_x = cv2.filter2D(img, cv2.CV_32F, sobel_x)
    grad_y = cv2.filter2D(img, cv2.CV_32F, sobel_y)
    magnitude = abs(grad_x) + abs(grad_y)
    img = np.uint8(np.clip(magnitude, 0, 255))
    return img

# 影像黑白化
def RGB_nagetive(img):

    # 先轉灰階
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    nr, nc = img.shape[:2]

    for x in range(nr):
        for y in range(nc):
            img[x, y] = 255 - img[x, y]
            # 負片轉換
            # 主要原因是因為影像梯度運算完後是黑底白線
            # 跟我們習慣的白底黑線不一樣，所以我把顏色做負片處理
            
            if img[x, y] < 200:
                img[x, y] = 0
            else:
                img[x, y] = 255
            # 類似二值化的概念(在教二值化之前做的，所以會有點不一樣)
            # 主要就是讓圖片只剩黑(0)白(255)兩個顏色

    return img

# 讀圖轉色
img = cv2.imread("test.jpg", -1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 影像梯度運算
# 再用梯度運算的結果去黑白化
img_S = sobel_gradient(img)
img_N = RGB_nagetive(img_S)

# 先做高斯濾波再運算
img_G = cv2.GaussianBlur(img, (5, 5), 0)
img_GS = sobel_gradient(img_G)
img_GN = RGB_nagetive(img_GS)

image = [img, img_S, img_N, img_G, img_GS, img_GN]
title = ['Original', 'S', 'N', 'G', 'GS', 'GN']

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(image[0+i], cmap='gray')
    plt.title(title[0+i])

plt.tight_layout()
plt.show()
