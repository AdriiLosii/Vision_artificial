import cv2
import numpy as np

filename = "../../data/images/salt-and-pepper.png"

img = cv2.imread(filename)
# Comprobamos se se leu a imaxe
if img is None:
    print("Non poiden ler a imaxe")

# Definimos o tamaño do kernel
kernelSize = 5

# Executamos o filtrado da mediana e o almacenamos nun array de numpy
medianBlurred = cv2.medianBlur(img,kernelSize)

cv2.imshow("Imaxe Orixinal", img)
cv2.waitKey(0)
cv2.imshow("Filtro da mediana : KernelSize = 5", medianBlurred)
cv2.waitKey(0)

