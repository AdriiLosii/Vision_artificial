# USA:
# python 3-APLICACION.py --first ../dataset/jp_01.png --second ../dataset/jp_02.png --detector SIFT --extractor SIFT

# importamos paquetes necesarios
from __future__ import print_function
import numpy as np
import argparse
import cv2
from imutils.feature.factories import FeatureDetector_create, DescriptorExtractor_create, DescriptorMatcher_create

# analizamos de argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--first", required=True, help="Path a primeira imaxe")
ap.add_argument("-s", "--second", required=True, help="Path a segunda imaxe")
ap.add_argument("-d", "--detector", type=str, default="SIFT",
	help="Detector de puntos clave a usar. "
		 "Opcions ['BRISK', 'DENSE', 'DOG', 'SIFT', 'FAST', 'FASTHESSIAN', 'SURF', 'GFTT', 'HARRIS', 'MSER', 'ORB', 'STAR']")
ap.add_argument("-e", "--extractor", type=str, default="SIFT",
	help="Extractor de caracteristicas a usar. Opcions ['SIFT', 'SURF']")
ap.add_argument("-m", "--matcher", type=str, default="BruteForce",
	help="Clasificador a usar para o match. Options ['BruteForce', 'BruteForce-SL2', 'BruteForce-L1', 'FlannBased']")
ap.add_argument("-v", "--visualize", type=str, default="Yes",
	help="Queres visualizar os resultados. Opcions ['Yes', 'No', 'Each']")
args = vars(ap.parse_args())

# Inicializamos os detectores de puntos clave
# Seo usuario elixe "DOG" ou "FASTHESSIAN", empregamos os valores axeitados
if args["detector"] == "DOG":
	detector = FeatureDetector_create("SIFT")
elif args["detector"] == "FASTHESSIAN":
	detector = FeatureDetector_create("SURF")
else:
	detector = FeatureDetector_create(args["detector"])

# incicalizamos o extractor de caracteristicas
extractor = DescriptorExtractor_create(args["extractor"])

# inicializamos o matches entre vectores de caracteristicas
matcher = DescriptorMatcher_create(args["matcher"])

# cargamos as duas imaxe a comparar
imageA = cv2.imread(args["first"])
imageB = cv2.imread(args["second"])
grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

# detectamos os puntos clavenas duas imaxes
kpsA = detector.detect(grayA)
kpsB = detector.detect(grayB)

# extraemos as caracteristicas locais en cada punto clave
(kpsA, featuresA) = extractor.compute(grayA, kpsA)
(kpsB, featuresB) = extractor.compute(grayB, kpsB)

# match os puntos calve empregando a distancia Euclidean entre os vectores de caracteristicas
rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
matches = []

if rawMatches is not None:
	# lazo sobre todos os matchs
	for m in rawMatches:
		# aseguramonos que a distancia pasa o limiar de David Lowe
		if len(m) == 2 and m[0].distance < m[1].distance * 0.8:
			matches.append((m[0].trainIdx, m[0].queryIdx))

	# Amosamos informacion para o diagnostico
	print("# de keypoints da primeira imaxe: {}".format(len(kpsA)))
	print("# de keypoints da segunda imaxe: {}".format(len(kpsB)))
	print("# de keypoints igualados: {}".format(len(matches)))

	#visualizamos
	(hA, wA) = imageA.shape[:2]
	(hB, wB) = imageB.shape[:2]
	vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
	vis[0:hA, 0:wA] = imageA
	vis[0:hB, wA:] = imageB

	# lazo sobre os matches
	for (trainIdx, queryIdx) in matches:
		# xeramos unha cor aleatorio e pintamos o match
		color = np.random.randint(0, high=255, size=(3,))
		color = tuple(map(int, color))
		ptA = (int(kpsA[queryIdx].pt[0]), int(kpsA[queryIdx].pt[1]))
		ptB = (int(kpsB[trainIdx].pt[0] + wA), int(kpsB[trainIdx].pt[1]))
		cv2.line(vis, ptA, ptB, color, 2)

		# debemos visualizar todos os matchs?
		if args["visualize"] == "Each":
			cv2.imshow("Matched", vis)
			cv2.waitKey(0)

	# Visualizacion
	if args["visualize"] == "Yes":
		cv2.imshow("Matched", vis)
		cv2.waitKey(0)
