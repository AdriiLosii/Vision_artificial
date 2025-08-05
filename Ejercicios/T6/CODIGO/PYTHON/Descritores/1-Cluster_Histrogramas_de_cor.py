# USA
# python cluster_histogramas_cor.py --dataset dataset

# Importamos os paquetes necesarios
from ClaseLabHistogram.labhistogram import LabHistogram
from sklearn.cluster import KMeans
from imutils import paths  #pip install imutils
import numpy as np
import argparse
import cv2

# argumentos de entrada e analizamolos
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path ao directorio do dataset de imaxes")
ap.add_argument("-k", "--clusters", type=int, default=2,
	help="# de cluster que desexamos")
args = vars(ap.parse_args())

# Creamos o obxecto que describe a imaxe por un histograma
# de cor e fixamos o numero de bis desexado
desc = LabHistogram([8, 8, 8])
data = []

# almacenamos os path das imaxes do directorio dataset
imagePaths = list(paths.list_images(args["dataset"]))
imagePaths = np.array(sorted(imagePaths))

# lazo sobre todas as imaxes do dataser
for imagePath in imagePaths:
	# cargamos a imaxe, describimola e metemola na lista
	image = cv2.imread(imagePath)
	hist = desc.describe(image)
	data.append(hist)

# Facemos un clustering sobre os histogramas de cor
clt = KMeans(n_clusters=args["clusters"], random_state=42)
labels = clt.fit_predict(data)

# lazo sobre as equiquetas unicas do array e ordenadas (np.unique)
for label in np.unique(labels):
	#almacenamos os path de todas as imaxes que son asocuiadas coa ultima etiqueta
	labelPaths = imagePaths[np.where(labels == label)]

	# lazo sobre todas as imaxes que pertencen a ultima etiqueta
	for (i, path) in enumerate(labelPaths):
		# cargamos a imaxe as as visualizamos
		image = cv2.imread(path)
		cv2.imshow("Cluster {}, Imaxe #{}".format(label + 1, i + 1), image)

	# Esperamos a que se presione unha tecla e  saimos
	cv2.waitKey(0)
	cv2.destroyAllWindows()