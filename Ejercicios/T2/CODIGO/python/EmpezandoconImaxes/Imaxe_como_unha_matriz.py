# Importamos as librerias
import cv2
import numpy as np

imagePath = "../data/images/number_zero.jpg"

# Lemos a imaxe en formato de grises
testImage = cv2.imread(imagePath,0)
print(testImage)

#Propiedades de imaxes
print("Tipo de datos = {}\n".format(testImage.dtype))
print("Tipo de obxecto = {}\n".format(type(testImage)))
print("Dimension da imaxe = {}\n".format(testImage.shape))
