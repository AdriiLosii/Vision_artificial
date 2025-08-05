# USA
# python 5-desc_Haralick_Textura.py --training ./dataset/training --test ./dataset/testing

# importamos os paquetes necesarios
from sklearn.svm import LinearSVC
import argparse
import mahotas #pip install mahotas==1.4.11 
import glob
import cv2

# argumentos de entrada e analizamolos
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--training", required=True, help="Path ao dataset de textures")
ap.add_argument("-t", "--test", required=True, help="Path as imaxes test")
args = vars(ap.parse_args())

# inicizalamos a matriz de datos e a lista de etiquetas
print("[INFO] extraendo as caracteristicas de textura...")
data = []
labels = []

# lazo sobre todas as imaxes de entrenamento
for imagePath in glob.glob(args["training"] + "/*.png"):
	# cargamos a imaxe, convertemola a gris e extraemos o nome da textura
	# a partir do nome do ficheiro
	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	texture = imagePath[imagePath.rfind("/") + 1:].split("_")[0]

	# Extremos as caracteristicas de  Haralick en 4 direccions, 
	#logo tomamos a media de cada dirección
	features = mahotas.features.haralick(image).mean(axis=0)

	# actualizamos os datos e as etiquetas
	data.append(features)
	labels.append(texture)

# entrenamos o clasificador
print("[INFO] entrenando o modelo...")
model = LinearSVC(C=10.0, random_state=42)
model.fit(data, labels)
print("[INFO] clasificando...")

# lazo sobre as imaxes de test
for imagePath in glob.glob(args["test"] + "/*.png"):
	# cargamos a imaxe, convertemola a gris e extraemos o descriptor de Haralick
	# de cada imaxe test
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	features = mahotas.features.haralick(gray).mean(axis=0)

	# clasificamos a imaxe test
	pred = model.predict(features.reshape(1, -1))[0]
	cv2.putText(image, pred, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
		(0, 255, 0), 3)

	# Visualiamos os resultados
	cv2.imshow("Imaxe", image)
	cv2.waitKey(0)