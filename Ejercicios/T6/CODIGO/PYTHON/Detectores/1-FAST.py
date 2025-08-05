# EMPREGA
# python FAST.py

# importamos os paquetes precisos
from __future__ import print_function
import numpy as np
import cv2
import imutils

# carga a imaxe a pasamola a gris
image = cv2.imread("../data/next.png")
orig = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#detectos FAST para opencv > 3
detector = cv2.FastFeatureDetector_create()
kps = detector.detect(gray, None)

print("# de keypoints: {}".format(len(kps)))

#Lazo sobre os keypoint e os anotamos na imaxe
for kp in kps:
	r = int(0.5 * kp.size)
	(x, y) = np.int0(kp.pt)
	cv2.circle(image, (x, y), r, (0, 255, 255), 2)

# Visualiamos a imaxe
cv2.imshow("Imaxe", np.hstack([orig, image]))
cv2.waitKey(0)