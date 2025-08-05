import cv2
import numpy as np

filename = "../../data/images/gaussian-noise.png"

img = cv2.imread(filename)
# Comprobamos se se leu a imaxe
if img is None:
    print("Non poiden ler a imaxe")

# Suavizado gaussiano
dst1=cv2.GaussianBlur(img,(5,5),0,0)
dst2=cv2.GaussianBlur(img,(25,25),50,50)

lineType=4
fontScale=1

# Visualizamos as imaxes
cv2.imshow("Imaxe Orixinal", img)
cv2.waitKey(0)
cv2.imshow("Suavizado gaussiano 1 : KernelSize = 5", dst1)
cv2.waitKey(0)
cv2.imshow("Suavizado gaussiano 2 : KernelSize = 25", dst2)
cv2.waitKey(0)


