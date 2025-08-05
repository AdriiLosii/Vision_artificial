import cv2
import numpy as np

filename = "../../data/Contour.png"
image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Non poiden ler a imaxe")



# Sobel na direccion x
sobelx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
# Sobel na direccion y
sobely = cv2.Sobel(image,cv2.CV_32F,0,1)

# Normalizamos para visualizacion
cv2.normalize(sobelx,
                dst = sobelx,
                alpha = 0,
                beta = 1,
                norm_type = cv2.NORM_MINMAX,
                dtype = cv2.CV_32F)
cv2.normalize(sobely,
                dst = sobely,
                alpha = 0,
                beta = 1,
                norm_type = cv2.NORM_MINMAX,
                dtype = cv2.CV_32F)
cv2.imshow("Gradiente de Sobel X", sobelx)
cv2.imshow("Gradiente de Sobel  Y", sobely)
cv2.waitKey(0)


