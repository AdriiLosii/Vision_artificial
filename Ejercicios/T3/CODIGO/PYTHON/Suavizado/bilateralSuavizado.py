import cv2
import numpy as np

img = cv2.imread("../../data/images/gaussian-noise.png")

# Comprobamos se se leu a imaxe
if img is None:
    print("Non poiden ler a imaxe")

# diametro para considerar entorno da vecindade
dia=15

# Valores altos mestura cores distantes
# producindo areas de cores semi-iguais
sigmaColor=80

#Valores altos permiten a influencia dos pixels alonxados
sigmaSpace=80

#Aplicamos o filtro
result = cv2.bilateralFilter(img, dia, sigmaColor, sigmaSpace)

cv2.imshow("Imaxe Orixinal", img)
cv2.waitKey(0)
cv2.imshow("Resulado do Filtro Bilateral", result)
cv2.waitKey(0)
