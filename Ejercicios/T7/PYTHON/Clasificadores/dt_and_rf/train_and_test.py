# USA
# python train_and_test.py --dataset 4scenes

# importacion
from __future__ import print_function
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from imutils import paths
import numpy as np
import argparse
import mahotas
import cv2
import sklearn
from sklearn.model_selection import train_test_split

def describe(image):
	# Extraemos a media e a stdev de cada canle de cor no espazo HSV
	(means, stds) = cv2.meanStdDev(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
	colorStats = np.concatenate([means, stds]).flatten()

	# Extraemos os descritores de texturas de Haralick da imaxe
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	haralick = mahotas.features.haralick(gray).mean(axis=0)

	# devolvemos un vector concatenado cos descritores de cor e textura para a imaxe
	return np.hstack([colorStats, haralick])

# Analizamos os argumento de entreda
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path ao dataset de escenas con 8 categorias")
ap.add_argument("-f", "--forest", type=int, default=-1,
	help="se e maior que 0 empregamos RandomForest e senon arbores de decision")
args = vars(ap.parse_args())

# paths as imaxes e reservamos listas para as caracteristicas e as etiquetas
print("[INFO] extraendo as caracteristicas...")
imagePaths = sorted(paths.list_images(args["dataset"]))
labels = []
data = []

# Para todas as imaxes
for imagePath in imagePaths:
	#extraemos as etiquetas e cargamos a imaxe dende disco
	label = imagePath[imagePath.rfind("/") + 1:].split("_")[0]
	image = cv2.imread(imagePath)

	# extraemos as caracteristicas e actualizamos os acumuladores
	features = describe(image)
	labels.append(label)
	data.append(features)

# Construimos os conxuntos de entrenamento (75%) e test (25%) 
# dos datos totais existente no directorio
(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(data),
	np.array(labels), test_size=0.25, random_state=42)

# inicializamos o clasiicador (model) como arbore de decicion (dt)
model = DecisionTreeClassifier(random_state=84)

# comprobamos se seleccionou o usuario un Random Forest en vez dun dt
if args["forest"] > 0:
	model = RandomForestClassifier(n_estimators=20, random_state=42)

# entrenamos o clasificador
print("[INFO] adestrando o clasificador...")
model.fit(trainData, trainLabels)

# avaliamos o clasificador
print("[INFO] avaliando...")
predictions = model.predict(testData)
print(classification_report(testLabels, predictions))

# lazo sobre unha cantas imaxes aleatorias
for i in list(map(int, np.random.randint(0, high=len(imagePaths), size=(10,)))):
	# lemos a imaxe a a clasficamos
	imagePath = imagePaths[i]
	filename = imagePath[imagePath.rfind("/") + 1:]
	image = cv2.imread(imagePath)
	features = describe(image)
	prediction = model.predict(features.reshape(1, -1))[0]

	# Visualizamos a predicion
	print("[PREDICION] {}: {}".format(filename, prediction))
	cv2.putText(image, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
	cv2.imshow("Imaxe", image)
	cv2.waitKey(0)
