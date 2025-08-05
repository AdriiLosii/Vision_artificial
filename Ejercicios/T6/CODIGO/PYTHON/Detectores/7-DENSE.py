# USA
# python 7-DENSE.py

# importamos paquetes
from __future__ import print_function
import numpy as np
import argparse
import cv2

def dense(image, step, radius):
	# incilizamos a lista de keypoints
	kps = []

	# lazo sobre filas e columnas da imaxe, colle un `step`
	# en cada direccion
	for x in range(0, image.shape[1], step):
		for y in range(0, image.shape[0], step):
			# creamos un keypoint e engadimolo a lista de keypoints
			kps.append(cv2.KeyPoint(x, y, radius * 2))

	# retornamos o lista denda de keypoints
	return kps

#  contrunimos os argumento e o analizados dos mesmos
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--step", type=int, default=28, help="step (en pixeles) do detector dense")
args = vars(ap.parse_args())

# cargamos a imaxe e pasamola a gris
image = cv2.imread("../data/next.png")
if image is None:
	print("Imaxe non atopada")
orig = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# inicilalizamos a lista de keypoint e a tupla de radios
kps = []
radii = (4, 8, 12)

# detecamos os puntos clase de  Dense para OpenCV 3+
rawKps = dense(gray, args["step"], 1)

# lazo sobre os keypoint se refinar
for rawKp in rawKps:
	#lazo sobre os radios que imos a empregar
	for r in radii:
		# construimos manualmente un keypoit e enegadimolo a lista
		kp = cv2.KeyPoint(x=rawKp.pt[0], y=rawKp.pt[1], _size=r * 2)
		kps.append(kp)

# mostramos a informacion
print("# dense keypoints: {}".format(len(rawKps)))
print("# dense + multi-radios keypoints: {}".format(len(kps)))

# lazo sobre os keypoint e anotamolos cun circulo
for kp in kps:
	r = int(0.5 * kp.size)
	(x, y) = np.int0(kp.pt)
	cv2.circle(image, (x, y), r, (0, 255, 255), 1)

# Visualizamos
cv2.imshow("Imaxes", np.hstack([orig, image]))
cv2.waitKey(0)