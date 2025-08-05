import cv2
import numpy as np

img = cv2.imread("../../data/images/sample.jpg", cv2.IMREAD_GRAYSCALE)
# Comprobamos se se leu a imaxe
if img is None:
    print("Non poiden ler a imaxe")
    
kernelSize = 3

# Aplicamos a Laplaciana
img1 = cv2.GaussianBlur(img,(3,3),0,0)
laplacian = cv2.Laplacian(img1, cv2.CV_32F, ksize = kernelSize,
                            scale = 1, delta = 0)

# Normalizamos os resultados
cv2.normalize(laplacian,
                dst = laplacian,
                alpha = 0,
                beta = 1,
                norm_type = cv2.NORM_MINMAX,
                dtype = cv2.CV_32F)

cv2.imshow("Laplaciana", laplacian)
cv2.waitKey(0)
