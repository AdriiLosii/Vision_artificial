# USA
# python rec_hog.py --training car_logos --test test_images

# Importamos os paquetes necesarios
from sklearn.neighbors import KNeighborsClassifier
from skimage import exposure
from skimage import feature
from imutils import paths
import argparse
import imutils
import cv2

# argumentos da linha de entrada e os analizamos
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--training", required=True, help="Path a base de datos de entrenamento")
ap.add_argument("-t", "--test", required=True, help="Path a base de datos test")
args = vars(ap.parse_args())

# Inicializamos as listas de imaes e etiquetas
print("[INFO] extraendo as caracteristicas...")
data = []
labels = []

# lazo sobre todas as imaxes de entrenamento
for imagePath in paths.list_images(args["training"]):
	# extraemos o fabricante do coche a partir no nome da imaxe
	make = imagePath.split("/")[-2]

	# cargamos a imaxe, pasamola a gris e extraemos os contornos
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	edged = imutils.auto_canny(gray)

	# atopamos os contornos e nos quedamos cos mais longos
	# que presumiblemente eran os do logo do coche
	cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	c = max(cnts, key=cv2.contourArea)

	# extraemos o logo e redimensionamos a un tamanho canonico de ancho e alto
	(x, y, w, h) = cv2.boundingRect(c)
	logo = gray[y:y + h, x:x + w]
	logo = cv2.resize(logo, (200, 100))

	# extraemos o Histogram of Oriented Gradients do logo
	H = feature.hog(logo, orientations=9, pixels_per_cell=(10, 10),
		cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")

	# actualizamos as listas de caracteristicas e etiquetas
	data.append(H)
	labels.append(make)

# entrenamos un clasificados de k_nn (nearest neighbors)
print("[INFO] entrenando o clasificador...")
model = KNeighborsClassifier(n_neighbors=1)
model.fit(data, labels)
print("[INFO] evaluando...")

# Lazo sobre as imaxes test
for (i, imagePath) in enumerate(paths.list_images(args["test"])):
	# cargamos a imaxe, pasamola a gris e a redimensionamos ao tamanho canonico
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	logo = cv2.resize(gray, (200, 100))

	# extraemos o Histogram of Oriented Gradients da imaxe test e
	#predecimos o fabricante
	(H, hogImage) = feature.hog(logo, orientations=9, pixels_per_cell=(10, 10),
		cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1", visualize=True)
	pred = model.predict(H.reshape(1, -1))[0]

	# Visualizamos a imaxe HOG
	hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
	hogImage = hogImage.astype("uint8")
	cv2.imshow("HOG Imaxe #{}".format(i + 1), hogImage)

	# Escrimimos a predicion e a visualizamos
	cv2.putText(image, pred.title(), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
		(0, 255, 0), 3)
	cv2.imshow("Test Image #{}".format(i + 1), image)
	cv2.waitKey(0)