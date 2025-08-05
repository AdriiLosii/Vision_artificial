import cv2
import numpy as np
from dataPath import DATA_PATH

filename = DATA_PATH+"images/gaussian-noise.png"

# Load an image
img = cv2.imread(filename)

# Comprobamos se se leu a imaxe
if img is None:
    print("Non poiden ler a imaxe")

# Filtro da media de tamaño 3
dst1=cv2.blur(img,(3,3),(-1,-1))

# Filtro da media de tamaño 7
dst2=cv2.blur(img,(7,7),(-1,-1))

cv2.imshow("Imaxe Orixianl", img)
cv2.waitKey(0)
cv2.imshow("Filtro media 1 : KernelSize = 3", dst1)
cv2.waitKey(0)
cv2.imshow("Filtro media 2 : KernelSize = 7", dst2)
cv2.waitKey(0)


