# USAGE
# python 4-DOG.py

# importamos paquetes necesario
from __future__ import print_function
import numpy as np
import cv2

# cargamos a imaxe e pasamola a gris
image = cv2.imread("../data/next.png")
if image is None:
	print("Imaxe non atopada")
orig = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detectmos puntos clave mediante diferencias de gaussianas (DoG() para OpenCV > 3.x
# E a mesma estratexia que emprega SIFT para a deteccion incial de puntos clave
detector = cv2.SIFT_create()
(kps, _) = detector.detectAndCompute(gray, None)

print("# de keypoints: {}".format(len(kps)))

# Lazo sobre os puntos clave detectados e anotamolos cun circulo
for kp in kps:
	r = int(0.5 * kp.size)
	(x, y) = np.int0(kp.pt)
	cv2.circle(image, (x, y), r, (0, 255, 255), 2)

# Visualizacion
cv2.imshow("Images", np.hstack([orig, image]))
cv2.waitKey(0)