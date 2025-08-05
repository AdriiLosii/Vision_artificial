# Importamos paquetes necesarios
import cv2
import numpy as np


# Lemos a imaxe
image = cv2.imread("../../data/images/boy.jpg")

scalingFactor = 1/255.0

# Convertimos de unsigned int a floar
image = np.float32(image)
# escalamos os valores para que pertenzan ao intervalo [0,1]
image = image * scalingFactor

#convertimos de novo a unsigned int
image = image * (1.0/scalingFactor)
image = np.uint8(image)
