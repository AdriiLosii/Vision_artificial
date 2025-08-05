# USA
# python GFFT.py

# importamos paquetes necesarios
from __future__ import print_function
import numpy as np
import cv2


def gftt(gray, maxCorners=0, qualityLevel=0.01, minDistance=1,
	mask=None, blockSize=3, useHarrisDetector=False, k=0.04):
	# computamos os putnos GFTT segundo parametros aportados (OpenCV 3)
	kps = cv2.goodFeaturesToTrack(gray, maxCorners, qualityLevel,
		minDistance, mask=mask, blockSize=blockSize,
		useHarrisDetector=useHarrisDetector, k=k)

	# creamos e dovolvemos o obxecto `KeyPoint`
	return [cv2.KeyPoint(pt[0][0], pt[0][1], 3) for pt in kps]

# cargamos a imaxe e a convertemos a gris
image = cv2.imread("../data/next.png")
if image is None:
	print("Imaxe non atopada")
orig = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# puntos clave GFTT empregando OpenCV 3+
kps = gftt(gray)

# lazo sobre os puntos clave e visualizamolos
for kp in kps:
	r = int(0.5 * kp.size)
	(x, y) = np.int0(kp.pt)
	cv2.circle(image, (x, y), r, (0, 255, 255), 2)

print("# de keypoints: {}".format(len(kps)))

# amosamos as iamxes
cv2.imshow("Imaxes", np.hstack([orig, image]))
cv2.waitKey(0)