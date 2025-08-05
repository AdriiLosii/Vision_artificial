# USA
# python hu_moments.py

# importamos os paquetes necesarios
import cv2
import imutils

# cargamos a imaxe e pasamola a gris
image = cv2.imread("./dataset/planes.png")
if image is None:
	print("Imaxe non atopada")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Achamos os Hu Moments da imaxe enteira e amosamolas
moments = cv2.HuMoments(cv2.moments(image)).flatten()
print("Momentos orixinales: {}".format(moments))
cv2.imshow("Imaxe", image)
cv2.waitKey(0)

# Atopamos os contornos na imaxe orixinal
cnts = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# Lazo sobre cosa contorno
for (i, c) in enumerate(cnts):
	# Extraera a ROI e achar os momentos da rexión de interese
	(x, y, w, h) = cv2.boundingRect(c)
	roi = image[y:y + h, x:x + w]
	moments = cv2.HuMoments(cv2.moments(roi)).flatten()

	# amosamos os momento de  ROI
	print("MOMENTOS PARA O AVION #{}: {}".format(i + 1, moments))
	cv2.imshow("ROI #{}".format(i + 1), roi)
	cv2.waitKey(0)