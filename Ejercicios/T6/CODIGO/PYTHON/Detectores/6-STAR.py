# USA
# python STAR.py

# importamos os paquetes necesarios
from __future__ import print_function
import numpy as np
import cv2


# load the image and convert it to grayscale
image = cv2.imread("../data/next.png")
if image is None:
	print("Imaxe non atopada")
orig = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detector de puntos clave STAR para OpenCV > 3+
detector = cv2.xfeatures2d.StarDetector_create()
kps = detector.detect(gray)

print("# de keypoints: {}".format(len(kps)))

# Lazo sobre os puntos clave e anotamos un circulo
for kp in kps:
	r = int(0.5 * kp.size)
	(x, y) = np.int0(kp.pt)
	cv2.circle(image, (x, y), r, (0, 255, 255), 2)

# Visualiamos a imaxe
cv2.imshow("Imaxes", np.hstack([orig, image]))
cv2.waitKey(0)