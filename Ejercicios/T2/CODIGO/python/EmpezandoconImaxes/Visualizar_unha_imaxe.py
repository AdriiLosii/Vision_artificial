# Importamos as librerias
import cv2
import numpy as np

imagePath = "../data/images/boy_with_axes.png"

# Lemos a imaxe en formato de gris
testImage = cv2.imread(imagePath,0)

#Amosamos con imshow de opencv
cv2.namedWindow('Imaxe',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Imaxe', 600,600) #Se a imaxe e moi grande reescalamos a venta de visualizacion
cv2.imshow("Imaxe",testImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("./testImage.png",testImage)
