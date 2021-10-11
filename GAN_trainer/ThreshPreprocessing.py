from PIL import Image
#import matplotlib.pylab as plt
import numpy as np
import cv2

#from google.colab import drive
#drive.mount('/content/gdrive')



n = 30060


for k in range(n):
    r = cv2.imread(r'C:\Users\Hye-Young\Desktop\kb\snukb_dataset\train\images\{}.jpg'.format(k))
    gray =  cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)

    #plt.imshow(r)
    #plt.show()

    black = 0
    white = 0
    

    th = int(np.mean(gray))
    print(th)
    _, output = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)
    output = Image.fromarray(output)

    for i in range(0,output.size[0]): # x방향 탐색
        for j in range(0,output.size[1]): # y방향 탐색
            gray = output.getpixel((i,j)) # i,j 위치에서의 RGB 취득

            if (gray<1) : 
                rgb_a = 0
                black = black + 1
            else : 
                rgb_a = 255
                white = white + 1

            output.putpixel((i,j),rgb_a)

    if black > white :
        for i in range(0,output.size[0]): # x방향 탐색
            for j in range(0,output.size[1]): # y방향 탐색
                rgb_a = output.getpixel((i,j))            
                rgb_r = 255-rgb_a
                output.putpixel((i,j),rgb_r)         

    # plt.imshow(output, 'gray')
    # plt.show()

    output.save(r"C:\Users\Hye-Young\Desktop\kb\images_preprocessed\{}.jpg".format(k)) # 파일 위치랑 이름 저장