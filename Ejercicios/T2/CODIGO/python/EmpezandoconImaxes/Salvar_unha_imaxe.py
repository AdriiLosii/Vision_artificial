# Importamos librarias
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)
matplotlib.rcParams['image.cmap'] = 'gray'

imagePath =  "./data/images/number_zero.jpg"
# Lemos a imaxe en formato de grises
testImage = cv2.imread(imagePath,0)

#Salvamos a imaxe a disco
cv2.imwrite("./data/test.jpg",testImage)

