import cv2
import numpy as np

def thinnin(img): #debe unha imaxe uint8!!

    img1 = img.copy()
    # Construimos o elemento estructural
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

    #Creamos un array negro
    thin = np.zeros(img1.shape,dtype='uint8')

    # Lazo ata que a erosión sonsigua un conxunto baleiro
    while (cv2.countNonZero(img1)!=0):
        # Erosion
        erode = cv2.erode(img1,kernel)
        # Apertura sobre a imaxe erosionada
        opening = cv2.morphologyEx(erode,cv2.MORPH_OPEN,kernel)
        # restamos ambalas duas imaxes anteriores
        subset = erode - opening
        # Realizamos a Union dos conxuntos anteriores
        thin = cv2.bitwise_or(subset,thin)
        # Preparamos a imaxe erosionada para a seguinte iteracion
        img1 = erode.copy()
    
    cv2.imshow('orixinal',img)
    cv2.imshow('adelgazada',thin)
    cv2.waitKey(0)