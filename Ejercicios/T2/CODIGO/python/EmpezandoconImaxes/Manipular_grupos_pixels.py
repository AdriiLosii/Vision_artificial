# Importamos as librarias
import cv2
import numpy as np

imagePath = "./data/images/number_zero.jpg"

# Lemos a imaxe en formato de gris
testImage = cv2.imread(imagePath,0)

#accedemos a unha rexión
test_roi = testImage[0:2,0:4]
print("Matriz orixinal\n{}\n".format(testImage))
print("Rexion seleccionada\n{}\n".format(test_roi))

testImage[0:2,0:4] = 111
print("Matriz modificada\n{}\n".format(testImage))

