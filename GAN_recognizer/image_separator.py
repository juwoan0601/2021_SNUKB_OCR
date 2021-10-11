import cv2
import numpy as np

def separateImage(image, image_size):
    shapeOfImage = image_size
    unitLen = shapeOfImage[1]  # 이미지의 height
    numChar = int(shapeOfImage[0] / shapeOfImage[1])  # 이미지를 나눌 횟수
    bin = int(shapeOfImage[0] / 2)
    separatedImages = []
    if numChar >= 4:
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
        totalImage = np.concatenate((totalImage, img.detach().numpy()), axis=3)
    return totalImage
