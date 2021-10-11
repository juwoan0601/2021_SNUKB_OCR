# -*- coding: utf-8 -*-
def expendBBox(x,y,w,h,shapeOfImage,expendContourRatio):
    expended_x = int(x-w*expendContourRatio*0.5)
    expended_y = int(y-h*expendContourRatio*0.5)
    if expended_x < 0: expended_x = 0
    if expended_y < 0: expended_y = 0
    expended_w = int(w+w*expendContourRatio)
    expended_h = int(h+h*expendContourRatio)
    if (expended_x+expended_w) > (shapeOfImage[1]): expended_w = shapeOfImage[1] - expended_x
    if (expended_y+expended_h) > (shapeOfImage[0]): expended_h = shapeOfImage[0] - expended_y
    return expended_x,expended_y,expended_w,expended_h

def OCR_gmobean(RUN_STEPS=[1,2,3,4,5,6], data_dir="./images", result_dir="./result"):
    # Prepare result folder
    import os
    import time
    from tqdm import tqdm
    PATH_DATA = data_dir
    PATH_RESULT = result_dir
    PATH_BNW = result_dir+"/BNW/"
    PATH_GAN = result_dir+"/GAN/"
    PATH_CRAFT = result_dir+"/CRAFT/"
    PATH_BOX = result_dir+"/BOX/"
    PATH_TEXT = result_dir+"/TEXT/"
    PATH_CHAR = result_dir+"/CHAR/"
    
    if not os.path.isdir(PATH_RESULT): os.mkdir(PATH_RESULT)
    if not os.path.isdir(PATH_BNW): os.mkdir(PATH_BNW)
    if not os.path.isdir(PATH_GAN): os.mkdir(PATH_GAN)
    if not os.path.isdir(PATH_CRAFT): os.mkdir(PATH_CRAFT)
    if not os.path.isdir(PATH_BOX): os.mkdir(PATH_BOX)
    if not os.path.isdir(PATH_TEXT): os.mkdir(PATH_TEXT)
    if not os.path.isdir(PATH_CHAR): os.mkdir(PATH_CHAR)

    startTime = time.time()
    listOfImage = os.listdir(PATH_DATA)
    # STEP 1: Pre-Processing 1 - Make "black text with white background" image
    if 1 in RUN_STEPS:
        print("STEP1 START {0}s ================".format(int(time.time()-startTime)))
        from preprocessing import thresh_processing_dir
        thresh_processing_dir(PATH_DATA, PATH_BNW)
    # STEP 2: GAN image
    if 2 in RUN_STEPS:
        print("STEP2 START {0}s ================".format(int(time.time()-startTime)))
        import os
        GAN_MODEL_NAME = "gen_ep40_batch128_data30060_initHe.pt"
        os.chdir("./GAN_recognizer")
        os.system("python Test_separate.py --data_dir ../{0} --Img_num {1} --ImgSave_dir ../{2} --model_name {3}".format(PATH_BNW,len(listOfImage),PATH_GAN,GAN_MODEL_NAME))
        os.chdir("../")
    # STEP 3: Make heatmap image by CRAFT
    if 3 in RUN_STEPS:
        print("STEP3 START {0}s ================".format(int(time.time()-startTime)))
        import os
        CRAFT_MODEL_NAME = "craft_mlt_25k.pth"
        os.chdir("./CRAFT-pytorch")
        os.system("python test_myOCR.py --trained_model=./saved_model/{0} --test_folder=../{1} --result_folder ../{2}".format(CRAFT_MODEL_NAME,PATH_GAN, PATH_CRAFT))
        os.chdir("../")
    # STEP 4: Make characterwise bounding box and cutting characterwise
    if 4 in RUN_STEPS:
        print("STEP4 START {0}s ================".format(int(time.time()-startTime)))
        import cv2
        import os
        import numpy as np
        for imageName in tqdm(listOfImage, desc="STEP4 Characterising"):
            imagePath = PATH_GAN+"/"+imageName
            image = cv2.imread(imagePath)
            shapeOfImage = image.shape
            heatmapPath = PATH_CRAFT+"/res_"+imageName.rsplit('.',1)[0]+"_mask.jpg"
            heatmap = cv2.imread(heatmapPath)
            shapeOfHeatmap = heatmap.shape
            cutY = int(shapeOfHeatmap[1]/2)
            areaOfImage = shapeOfImage[0]*shapeOfImage[1]
            _characterRegionHeatmap = heatmap[:, 0:cutY].copy()
            characterRegionHeatmap = cv2.resize(_characterRegionHeatmap, dsize=(shapeOfImage[1], shapeOfImage[0]), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(characterRegionHeatmap, cv2.COLOR_BGR2GRAY)
            #thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,2)
            _, thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
            # STEP 2-1: Find contours
            cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            selected_cnt_xywh = []
            ratioWH = shapeOfImage[1]/shapeOfImage[0] # images width-height ratio. it will be help for how many character is
            for c in cnts:
                x,y,w,h = cv2.boundingRect(c)
                #if w*h > (1/(ratioWH*4))*areaOfImage:
                if w*h > areaOfImage*0.05:
                    cv2.rectangle(characterRegionHeatmap, (x, y), (x + w, y + h), (36,255,12), 2)
                    cv2.rectangle(thresh, (x, y), (x + w, y + h), (36,255,12), 2)
                    selected_cnt_xywh.append((x,y,w,h))
            contourPath = "{0}/{1}_contour.jpg".format(PATH_BOX,imageName.rsplit('.',1)[0])
            threshPath = "{0}/{1}_thresh.jpg".format(PATH_BOX,imageName.rsplit('.',1)[0])
            cv2.imwrite(contourPath, characterRegionHeatmap)
            cv2.imwrite(threshPath, thresh)
            # STEP 2-2: Cutting characterwise
            num_char = 0
            expendContourRatio = 0.3
            expendBorderRatio = 0.2
            sorted_cnt_xywh = sorted(selected_cnt_xywh, key=lambda xywh: xywh[0])   # sort by x value
            if not os.path.isdir("./{0}/{1}".format(PATH_RESULT,imageName.rsplit('.',1)[0])):
                os.mkdir("./{0}/{1}".format(PATH_RESULT,imageName.rsplit('.',1)[0]))
            for (x, y, w, h) in sorted_cnt_xywh:
                expended_x,expended_y,expended_w,expended_h = expendBBox(x, y, w, h,characterRegionHeatmap.shape,expendContourRatio)
                characterPath = "./{2}/{0}/{0}_C{1}.jpg".format(imageName.rsplit('.',1)[0], num_char, PATH_RESULT)
                num_char += 1
                characterImage = image[:, expended_x:expended_x+expended_w].copy() # vertical cutting
                characterImage = cv2.copyMakeBorder(characterImage, int(h*expendBorderRatio), int(h*expendBorderRatio), int(w*expendBorderRatio), int(w*expendBorderRatio), cv2.BORDER_CONSTANT, None, value = [255,255,255]) # add white border
                cv2.imwrite(characterPath, characterImage)
    # STEP 5: Character to text
    if 5 in RUN_STEPS:
        print("STEP5 START {0}s ================".format(int(time.time()-startTime)))
        # Assemble each char image file
        from distutils.dir_util import copy_tree
        for imageName in tqdm(listOfImage, desc="STEP5 Move Images"):
            copy_tree("./{0}/{1}".format(PATH_RESULT, imageName.rsplit('.',1)[0]), PATH_CHAR)
        import os
        CR_MODEL_NAME = "./saved_model/best_acc_0.99_model.h5"
        os.chdir("./hangul-syllable-recognition")
        os.system("python test_myOCR.py --test_data ../{0} --saved_model {1} --result_dir {2}".format(PATH_CHAR,CR_MODEL_NAME,PATH_TEXT))
        os.chdir("../")

    # STEP 6: Make result csv file
    if 6 in RUN_STEPS:
        print("STEP6 START {0}s ================".format(int(time.time()-startTime)))
        resultWordPath = "./{0}/result.csv".format(PATH_RESULT)
        resultWordFile = open(resultWordPath, 'w', encoding='utf-8')
        resultWordFile.write("index,label\n")

        resultTextPath = "./{0}/CHAR.txt".format(PATH_TEXT)
        resultTextFile = open(resultTextPath, 'r', encoding='utf-8')
        resultText = resultTextFile.readlines()
        resultTextFile.close()
        listOfCharImage = os.listdir(PATH_CHAR)
        wordDict = {}
        for imageName in listOfImage:
            wordDict[imageName.rsplit('.',1)[0]] = []
        for imageName in tqdm(listOfImage, "STEP6 Generate Word"):
            for charImageNum in range(len(listOfCharImage)):
                if imageName.rsplit('.',1)[0] == listOfCharImage[charImageNum].split('_')[0]:
                    wordDict[imageName.rsplit('.',1)[0]].append(resultText[charImageNum].replace('\n',''))
        
        for imageName in listOfImage:
            word = ''.join(wordDict[imageName.rsplit('.',1)[0]])
            resultWordFile.write('{0},{1}\n'.format(imageName.rsplit('.',1)[0],word))
        print("Result saved at {0}".format(resultWordPath))
    print('Predict {0} images at {1} sec. Average: {2} sec/image'.format(len(listOfImage), int(time.time()-startTime), (int(time.time()-startTime)/len(listOfImage))))

if __name__ == '__main__':
    import os
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_tag', type=str, default='test')
    parser.add_argument('--output_tag', type=str, default='test')
    parser.add_argument('--steps', type=int, nargs='+', default=[1,2,3,4,5,6], help='STEP index to run')
    opt = parser.parse_args()
    data_dir = "./images/{0}/".format(opt.input_tag)
    result_dir = "./result/{0}/".format(opt.input_tag+'_'+opt.output_tag)
    OCR_gmobean(RUN_STEPS=opt.steps, data_dir=data_dir, result_dir=result_dir)