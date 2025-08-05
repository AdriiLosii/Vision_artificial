# Importamos librarias
import cv2
import numpy as np

imagePath =  "./data/images/number_zero.jpg"

# Lemos a imaxe en formato de grises
testImage = cv2.imread(imagePath,0)

#accedemos ao primeiro pixel
print(testImage[0,0])

#Modificamos o valor deste pixel
testImage[0,0]=200
print(testImage)
