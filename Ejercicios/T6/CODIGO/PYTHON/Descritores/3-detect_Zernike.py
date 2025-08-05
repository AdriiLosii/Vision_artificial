# USA
# python 3-detect_Zernike.py

# importamos os paquetes necesarios
from scipy.spatial import distance as dist
import numpy as np
import mahotas #pip install mahotas==1.4.11
import cv2
import imutils

def describe_shapes(image):
	# Inicializa as listas das caracteristicas de forma
	shapeFeatures = []

	# convertimos a iamxes a gris, suavizamos e aplicamos o limiar
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (13, 13), 0)
	thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)[1]

	# realizamos unha serie de dilatacions para pechar ocos nas formas
	thresh = cv2.dilate(thresh, None, iterations=4)
	thresh = cv2.erode(thresh, None, iterations=2)

	# detectamos uns contornos
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	# lazo sobre todos os contornos
	for c in cnts:
		# creamos unha mascara e debuxamos os contornos
		mask = np.zeros(image.shape[:2], dtype="uint8")
		cv2.drawContours(mask, [c], -1, 255, -1)

		# extraemos a imaxe ROI dende a mascara
		(x, y, w, h) = cv2.boundingRect(c)
		roi = mask[y:y + h, x:x + w]

		# acha os momentos de Zernike para a ROI e actualzia o lista do descritores de forma
		features = mahotas.features.zernike_moments(roi, cv2.minEnclosingCircle(c)[1], degree=8)
		shapeFeatures.append(features)

	# devolvemos a tupla de contornos e descritores de forma
	return (cnts, shapeFeatures)

# Cargamos o template do obxecto que queremos detectar e atopamola
# na zona da rexion do xogo onde queremos atopala
refImage = cv2.imread("./dataset/pokemon_red.png")
if refImage is None:
	print("imaxe non atopada")
(_, gameFeatures) = describe_shapes(refImage)

# cargamos a imaxe e describimolas
shapesImage = cv2.imread("./dataset/shapes.png")
if shapesImage is None:
	print("imaxe non atopada")
(cnts, shapeFeatures) = describe_shapes(shapesImage)

# Distancias as distancias Euclidean entre o video do xogo e todas as formas
# na segunda imaxe e enton atopamos o indice de menos distanica
D = dist.cdist(gameFeatures, shapeFeatures)
i = np.argmin(D)

# lazo sobre os contonos nas formas da imaxe
for (j, c) in enumerate(cnts):
	# se o índice do contorno actual non e igual ao indice
	# contorno coa distancia máis pequena e despois debuxa
	# na imaxe de saída
	if i != j:
		box = cv2.minAreaRect(c)
		box = np.int0(cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box))
		cv2.drawContours(shapesImage, [box], -1, (0, 0, 255), 2)

# debuxamos unha caixa arredor do obxecto detectado
box = cv2.minAreaRect(cnts[i])
box = np.int0(cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box))
cv2.drawContours(shapesImage, [box], -1, (0, 255, 0), 2)
(x, y, w, h) = cv2.boundingRect(cnts[i])
cv2.putText(shapesImage, "ATOPADO!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
	(0, 255, 0), 3)

# visualizamos os resutlados
cv2.imshow("Imaxe de entrada", refImage)
cv2.imshow("Formas detectadas", shapesImage)
cv2.waitKey(0)
