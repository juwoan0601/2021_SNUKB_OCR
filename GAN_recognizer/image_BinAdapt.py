import cv2
import numpy as np
from PIL import Image


def separateImage(image, image_size):
    shapeOfImage = image_size
    unitLen = shapeOfImage[1]  # 이미지의 height
    numChar = int(shapeOfImage[0] / shapeOfImage[1] + 0.5)  # 이미지를 나눌 횟수
    separatedImages = []
    if numChar >= 4:
        bin = sep_point(image[0], image_size)
        temp = image[:, :, :, :bin]
        separatedImages.append(temp)
        temp = image[:, :, :, bin:]
        separatedImages.append(temp)

    else:
        separatedImages.append(image)

    return separatedImages


def mergeImage(images):
    imageList = images
    totalImage = imageList[0].detach().numpy()
    for img in imageList[1:]:
        totalImage = np.concatenate((totalImage[:, :, :, :-7], img.numpy()[:, :, :, 7:]), axis=3)
    return totalImage


def sep_point(image, image_size):
    image = np.reshape(image, (image_size[1], image_size[0], 3))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    black = 0
    white = 0
    th = np.mean(gray)
    _, output = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)

    for i in range(0, output.shape[0]):  # x방향 탐색
        for j in range(0, output.shape[1]):  # y방향 탐색
            gray = output[i][j]  # i,j 위치에서의 RGB 취득

            if (gray < 1):
                rgb_a = 0
                black = black + 1
            else:
                rgb_a = 1
                white = white + 1

            output[i][j] = rgb_a

    if black > white:
        for i in range(0, output.shape[0]):  # x방향 탐색
            for j in range(0, output.shape[1]):  # y방향 탐색
                rgb_a = output[i][j]  # i,j 위치에서의 RGB 취득
                rgb_r = 1 - rgb_a
                output[i][j] = rgb_r

    bin = int(image_size[0] / 2)
    left = 0
    right = 0
    Llist = []
    Rlist = []
    try:
        for i in range(50):
            val_1 = np.mean(output[:, bin - left])
            Llist.append(val_1)
            left += 1

        for i in range(50):
            val_2 = np.mean(output[:, bin + right])
            Rlist.append(val_2)
            right += 1

        val_1 = max(Llist)
        val_2 = max(Rlist)
        left = Llist.index(val_1)
        right = Rlist.index(val_2)

        if val_1 > val_2:
            return bin - left
        else:
            return bin + right

    except:
        return bin