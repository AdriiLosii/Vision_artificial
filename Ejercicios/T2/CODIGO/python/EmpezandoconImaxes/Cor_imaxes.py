# Importamos librarias
import cv2
import numpy as np

# Path a imaxe para ser cargada
imagePath = "./data/images/musk.jpg"

# Lemos a imaxe e imprimimos as canles
img = cv2.imread(imagePath)
print("Dimensions da imaxe ={}".format(img.shape))
