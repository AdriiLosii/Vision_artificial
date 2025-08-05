# USA
# python HARRIS.py

# importamos os modulos
from __future__ import print_function
import numpy as np
import cv2
#import imutils

def harris(gray, blockSize=2, apetureSize=3, k=0.1, T=0.02):
	# convertimos a imaxe de entrada a punto flotante e
	# logo achamos a matriz de Harris
	gray = np.float32(gray)
	H = cv2.cornerHarris(gray, blockSize, apetureSize, k)

	# recuperamos as coordenadas (x, y) onde o valor de Harris e superior a un
	# limiar (punto clave), considerando un entorno de radio 3
	kps = np.argwhere(H > T * H.max())
	kps = [cv2.KeyPoint(pt[1], pt[0], 3) for pt in kps]

	# Devolvemos os puntos clave detectados
	return kps

# cargamos a imaxe e pasamola a gris
image = cv2.imread("../data/next.png")
if image is None:
	print("imaxe non atopada")
orig = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# A funcion cornerHarris require un array de tipo float32
gray = np.float32(gray)

# Detector de Harris
harris_corners = cv2.cornerHarris(gray, 3, 3, 0.05)
#obtemos as coordenadas das esquinas e pintamos un pixel ou un circulo
#un pixel soamente en cor verde
#image[harris_corners > 0.025 * harris_corners.max() ] = [255, 0, 0]
#pintamos un circulo amarelo de radio 4
r,c = np.where(harris_corners > 0.025 * harris_corners.max())
for point in zip(c,r):
    cv2.circle(image,point, 4, (0, 255, 255), 2)
    
# Visualiamos a imaxe
cv2.imshow("Imaxe", image)
cv2.waitKey(0)