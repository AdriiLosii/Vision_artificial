import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['figure.figsize'] = (10.0, 10.0)
matplotlib.rcParams['image.cmap'] = 'gray'

# Cargamos la imagen
image_path = '/home/adrian/Escritorio/Robotica/3º/3_Curso_1_Cuatri/Vision_Artificial/Practicas/P2/Exercicios_clase/flowers.png'
image = cv2.imread(image_path, 1)
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Por si no se encuentra la imagen
if image is None:
    print('No se encuentra la imagen')
    exit(1)

# Visualizamos la imagen original
cv2.namedWindow('Flores original',cv2.WINDOW_NORMAL)
cv2.imshow('Flores original',image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Modificamos los colores
h, s, v = cv2.split(image_hsv)
merge_hsv_new = cv2.merge((h + 20, s, v))
image_bgr_mod = cv2.cvtColor(merge_hsv_new, cv2.COLOR_HSV2BGR)

# Visualizamos la imagen
cv2.namedWindow('Flores modificadas',cv2.WINDOW_NORMAL)
cv2.imshow('Flores modificadas',image_bgr_mod)
cv2.waitKey(0)
cv2.destroyAllWindows()
