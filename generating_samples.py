
import cv2
import numpy as np
import mahotas

from PIL import Image
from PIL import ImageEnhance
xi=90
digit=1
rois=50
cont=1
size=0
import os
path, dirs, files = next(os.walk("./images/"))
images_count = len(files)
#LOAING IMAGE-----------------------------------------------------------------------------------------------------------

for filename in os.listdir('./images/'):
    print filename
    cont1,cont2= filename.split("img")
    cont2, cont3=  cont2.split(".")
    cont=cont2
"""
for cont in range(30,34,1):#change the new picture in the folder images
"""
    xf=1
    xfx=xf
    image = cv2.imread('images/'+'img'+str(cont)+'.bmp')
    image = cv2.resize(image, (400, 250))
    #--------detecting dial ---------
    sample = cv2.imread("sample_big.jpg")
    sample_h, sample_w, sample_k = sample.shape
    h, w, k = image.shape
    res = cv2.matchTemplate(image,sample,cv2.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    (x,y) = min_loc
    print "CONT", cont
    #print(x, y)
    #print "loc", max_loc[0], max_loc[1]
    x_center = max_loc[0] + sample_w/2
    y_center = max_loc[1] + sample_h/2

    print(x_center, y_center)
    """
    cv2.line(image, (x_center, y_center-1), (x_center, y_center+1),(255,0,0),3)

    cv2.line(image, (0, y_center), (w, y_center),(0,255,0),4)

    cv2.line(image, (0, y_center-20), (w, y_center-20),(255,0,0),3)

    cv2.line(image, (0, y_center+20), (w, y_center+20),(255,0,0),3)

    cv2.imshow("line", image)

    cv2.waitKey(0)
    """
    if cont<23:
        rois=rois
        size=6
    else:
        rois=y_center-20
        size=9
    #------------------------------------
    def black_and_white(color_image):
       bw = color_image.convert('L')
       return bw
    cv2.imshow("bw",np.asarray(black_and_white(Image.fromarray(Image.fromarray(image)))))
    def adjust_contrast(input_image, factor):
       # image1 = Image.open(input_image)
        enhancer_object = ImageEnhance.Contrast(input_image)
        out = enhancer_object.enhance(factor)
        return out

    gris = cv2.cvtColor(np.asarray(adjust_contrast(image, 2.1)), cv2.COLOR_BGR2GRAY)

    mascar=np.zeros(image.shape[:2], dtype="uint8")
    #cv2.imshow("image", mascar)

    cv2.rectangle(mascar, (xf, rois), (xf+400, rois+50), 255, -1)
    image2=cv2.bitwise_and(gris,gris,mask=mascar)


    cv2.imshow("image1", image2)
    cv2.waitKey(0)
    T3=mahotas.thresholding.otsu(image2)
    gris_copy=gris.copy()
    gris_2=gris.copy()
    #NEGATIVE IMAGE---------------------------------------------------------------------------------------------------------
    for j in range(1,400,1):
        for i in range(1,250,1):
            color=gris[i,j]
            gris[i,j]=255-color
    gris=cv2.GaussianBlur(gris, (3, 3),0)
    #cv2.imshow("image4", gris)
    T1=mahotas.thresholding.otsu(gris)
    clahe = cv2. createCLAHE(clipLimit=1.0)
    grises= clahe . apply(gris)
    conteo=1
    T2 = mahotas.thresholding.otsu(grises)
    T=(T2+T1+5)/2



    #THRESHOLD--------------------------------------------------------------------------------------------------------------
    for k in range(rois,rois+45,1):
        for z in range(xf,400,1):
            color=grises[k,z]
            if color>T:
                grises[k,z]=0
            else:
                grises[k,z]=255
    #cv2.imshow("gris",grises)
    def adjust_contrast(input_image, factor):
       # image1 = Image.open(input_image)
        enhancer_object = ImageEnhance.Contrast(Image.fromarray(input_image))
        out = enhancer_object.enhance(factor)
        return out
    grises = np.asarray(adjust_contrast(grises, 2.1))
    cv2.imshow("MEDIDOR",grises)
    cv2.waitKey(0)


    #MASCARA FOR ROI--------------------------------------------------------------------------------------------------------
    mascara=np.zeros(image.shape[:2], dtype="uint8")
    cv2.rectangle(mascara, (xf, rois), (xf+400, rois+50), 255, -1)
    image1=cv2.bitwise_and(grises,grises,mask=mascara)
    cv2.imshow("Med",image1)
    cv2.waitKey(0)
    #FILTER-----------------------------------------------------------------------------------------------------------------
    blurred = cv2.GaussianBlur(image1, (7,7),0)
    blurred = cv2.medianBlur(blurred,1)

    #THRESHOLD--------------------------------------------------------------------------------------------------------------
    v = np.mean(blurred)
    sigma=0.33
    lower = (int(max(0, (1.0 - sigma) * v)))
    upper = (int(min(255, (1.0 + sigma) * v)))
    #EDGE DETECTION---------------------------------------------------------------------------------------------------------
    edged = cv2.Canny(blurred, lower, upper)
    #cv2.imshow("edged",edged)
    (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnts = sorted([(c, cv2.boundingRect(c)[0]) for c in cnts], key = lambda x: x[1])
    yf=rois
    vec=[]

#EDGE DETECTION---------------------------------------------------------------------------------------------------------
    edged = cv2.Canny(blurred, lower, upper)
    cv2.imshow("gris",edged)

    (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnts = sorted([(c, cv2.boundingRect(c)[0]) for c in cnts], key = lambda x: x[1])
    yf=rois
    vec=[]
    digit2=1
#EDGE RECOGNITION-------------------------------------------------------------------------------------------------------
    consumo=0
    for (c,_) in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if w > 7 and h > 11 and w<30:
          if(x-xfx)>10:
            if digit2<size:
                print digit
                xfx=x+w
                yf=y
                roi=gris[y:y+h,x:x+w]
                guardar=roi.copy()

                cv2.imshow("roi",roi)
                cv2.waitKey(0)

                cv2.imwrite(("samples/"+str(digit)+"_"+str(cont)+'.png'),guardar)
                digit2+=1
                digit+=1
