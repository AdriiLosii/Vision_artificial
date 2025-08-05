# USA
# python 8-BRISK.py

# importamos os paquetes
from __future__ import print_function
import numpy as np
import cv2

# cargamos a imaxe e pasamola a gris
image = cv2.imread("../data/next.png")
if image is None:
	print("Imaxe non atopada")
orig = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#  BRISK para OpenCV 3+
detector = cv2.BRISK_create()
kps = detector.detect(gray, None)
print("# de keypoints: {}".format(len(kps)))

# lazo sobre puntos de interes e os debuxamos
for kp in kps:
	r = int(0.5 * kp.size)
	(x, y) = np.int0(kp.pt)
	cv2.circle(image, (x, y), r, (0, 255, 255), 2)

# Visualizamos
cv2.imshow("Imaxes", np.hstack([orig, image]))
cv2.waitKey(0)