# USA:
# python 1-SIFT.py --image ../dataset/jp_01.png

# importamos paquetes
from __future__ import print_function
import argparse
import cv2
import imutils

# linha de argumentos e o seu analise
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path a imaxe de entrada")
args = vars(ap.parse_args())

# cargamos a imaxe e pasamola a gris
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Inicializamos o detector para opencv 3+
# detectamos os puntos de interese e achamos sobre eles os descritores SIFT
detector = cv2.SIFT_create()
(kps, descs) = detector.detectAndCompute(gray, None)

# Visualizamos os puntos detectados e a forma dos vectores de caracteristicas
print("[INFO] # de keypoints detectados: {}".format(len(kps)))
print("[INFO] dimensions do vector de caracteristicas: {}".format(descs.shape))