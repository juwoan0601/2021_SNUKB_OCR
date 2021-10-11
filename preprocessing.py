
def thresh_processing(image_path, result_path):
    import numpy as np
    import cv2
    from PIL import Image

    r = cv2.imread(image_path)
    gray =  cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)

    black = 0
    white = 0

    th = int(np.mean(gray))
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

    output.save(result_path) # 파일 위치랑 이름 저장

def thresh_processing_dir(image_dir, result_dir):
    import os
    from tqdm import tqdm
    image_list = os.listdir(image_dir)
    for image_name in tqdm(image_list, desc='Image List'):
        thresh_processing("{0}/{1}".format(image_dir,image_name), "{0}/{1}".format(result_dir,image_name))
    return 0

if __name__ == "__main__":
    thresh_processing_dir("./images/test", "./th")