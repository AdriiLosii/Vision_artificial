# Importamos as librerias
import cv2
import numpy as np

imagePath = "../data/gaiteiro.jpg"

# Lemos a imaxe en formato de gris
testImage = cv2.imread(imagePath,0)
if testImage is None:
    print('Non se atopa a imaxe {}'.format(imagePath))
    exit(1)

#Escribimos a imaxe a disco
cv2.imwrite("../data/gaiteiro_gris.png",testImage)

#Amosamos a imaxe con imshow de opencv
cv2.namedWindow('Imaxe',cv2.WINDOW_NORMAL)
cv2.imshow('Imaxe',testImage)
cv2.waitKey(0)
cv2.destroyAllWindows()

