# Importamos paquetes necesarios
import cv2
import numpy as np

# Lemos a imaxe
image = cv2.imread("../../data/images/boy.jpg")

brightnessOffset = 50

# Sumamos un offser para incrementar o brilo
brightHigh = image + brightnessOffset

# Visualizamos as imaxe
cv2.imshow("Orixinal",image)
cv2.imshow("Alto brilo",brightHigh)
cv2.waitKey(0)
cv2.imwrite("../../data/images/briloAlto.png",brightHigh)

print("Tipo de datos da Imaxe Orixinal : {}".format(image.dtype))
print("Tipo de datos da Imaxe alto brilo : {}\n".format(brightHigh.dtype))

print("Maximo da imaxe orixinal : {}".format(image.max()))
print("Maximo da imaxe de alto brilo : {}".format(brightHigh.max()))
