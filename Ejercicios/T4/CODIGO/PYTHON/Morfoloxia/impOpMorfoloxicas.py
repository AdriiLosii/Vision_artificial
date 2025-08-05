import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)
matplotlib.rcParams['image.cmap'] = 'gray'

im = np.zeros((10,10),dtype='uint8')
print(im)
cv2.imshow("imaxe",im*255)
cv2.waitKey(0)

im[0,1] = 1
im[-1,0]= 1
im[-2,-1]=1
im[2,2] = 1
im[5:8,5:8] = 1

print(im)
cv2.imshow("imaxe",255*im)
cv2.waitKey(0)

element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
print(element)

ksize = element.shape[0]
height,width = im.shape[:2]

border = ksize//2
# Creamos unha imaxe padeada con ceros
paddedIm = np.zeros((height + border*2, width + border*2))
paddedIm = cv2.copyMakeBorder(im, border, border, border, border, cv2.BORDER_CONSTANT, value = 0)
for h_i in range(border, height+border):
    for w_i in range(border,width+border):
        # Buscamos un pixel branco
        if im[h_i-border,w_i-border]:

            paddedIm[ h_i - border : (h_i + border)+1, w_i - border : (w_i + border)+1] = \
                cv2.bitwise_or(paddedIm[ h_i - border : (h_i + border)+1, w_i - border : (w_i + border)+1],element)

            # Visualizamos o resultado
            #print(paddedIm)
            cv2.imshow("imaxe",255*paddedIm)
            cv2.waitKey(0)

dilatedEllipseKernel = cv2.dilate(im, element)
print(dilatedEllipseKernel)
cv2.imshow("imaxe",255*dilatedEllipseKernel)
cv2.waitKey(0)


