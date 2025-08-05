# USA
# python matchMultScale.py --template ../../data/cod_logo.png --images ../../data/imaxes_TM

# Importamos
import numpy as np
import argparse
import glob
import cv2


def resize(image, width = None, height = None, inter = cv2.INTER_AREA):
	# iinicializamos as dimensions da imaxe a ser redimensionada
	dim = None
	(h, w) = image.shape[:2]

	# Se tanto width e height son None, enton retornamos a imaxe orixnal
	if width is None and height is None:
		return image

	# Vemos se width e None
	if width is None:
		# Achamos a ratio entre as height e formamos as dimension desexadas
		r = height / float(h)
		dim = (int(w * r), height)

	# Doutra forma height é None
	else:
		# Achamos a ratio entre as e formamos as dimension desexadas
		r = width / float(w)
		dim = (width, int(h * r))

	# redimensionamos a imaxe
	resized = cv2.resize(image, dim, interpolation = inter)

	# devolvemos a imaxe desexada
	return resized




# Analizamos a linha de comando
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--template", required=True, help="Path ao template")
ap.add_argument("-i", "--images", required=True,
	help="Path as imaxe onde verificaremos se esta o template")
ap.add_argument("-v", "--visualize",
	help="Flag para visualizar resultados intermedios nas escalas")
args = vars(ap.parse_args())

# Cargamos imaxe, gris --> deteccion de bordes
template = cv2.imread(args["template"])
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
template = cv2.Canny(template, 50, 200)
(tH, tW) = template.shape[:2]
cv2.imshow("Template", template)

# Lazo sobre todas as imaxe 
for imagePath in glob.glob(args["images"] + "/*.jpg"):
	# Cargamos a imaxe--> gris e mantemos a pista na variable
	# bookkeeping a pista da rexion on hao matched
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	found = None

	# Lazo sobre as escalas
	for scale in np.linspace(0.2, 1.0, 20)[::-1]:
		# Redimensionamos acorde a escala, e mantemos a pista da ratio do resize
		resized = resize(gray, width = int(gray.shape[1] * scale))
		r = gray.shape[1] / float(resized.shape[1])

		# Se a imaxe redimensionada e mais pequena que o template, rompemos o lazo
		if resized.shape[0] < tH or resized.shape[1] < tW:
			break

		# Bordes e aplicamos o template matching
		edged = cv2.Canny(resized, 50, 200)
		result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
		(_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

		# Visualizamos resultados intermedios
		if args.get("visualize", False):
			# debuxamos un rectangulo na rexion que produce o match
			clone = np.dstack([edged, edged, edged])
			cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
				(maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
			cv2.imshow("Visualiza", clone)
			cv2.waitKey(0)

		# Se atopamos un novo maximo na variable de correlacion, enton
		# actualziamos a variable bookkeeping
		if found is None or maxVal > found[0]:
			found = (maxVal, maxLoc, r)

	# Desempaquetamos a variable bookkeeping e achamos as cooordenadas (x, y) 
	# da caixa envolvente e o tamanho
	(_, maxLoc, r) = found
	(startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
	(endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

	# Visualziamos
	cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
	cv2.imshow("Imaxe", image)
	cv2.waitKey(0)