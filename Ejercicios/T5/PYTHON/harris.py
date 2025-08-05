from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
source_window = 'Imaxe de entrada'
corners_window = 'Esquinas detectadas'
max_thresh = 255
def cornerHarris_demo(val):
    thresh = val
    # Parametro do detector
    blockSize = 2
    apertureSize = 3
    k = 0.04
    # Detectando as esquinas
    dst = cv.cornerHarris(src_gray, blockSize, apertureSize, k)
    # Normalizamos
    dst_norm = np.empty(dst.shape, dtype=np.float32)
    cv.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    dst_norm_scaled = cv.convertScaleAbs(dst_norm)
    # Debuxamos un circulo arredor das esquinas
    for i in range(dst_norm.shape[0]):
        for j in range(dst_norm.shape[1]):
            if int(dst_norm[i,j]) > thresh:
                cv.circle(dst_norm_scaled, (j,i), 5, (0), 2)
    # Visualizamos o resultados
    cv.namedWindow(corners_window)
    cv.imshow(corners_window, dst_norm_scaled)
# Cargamos a imaxe fonte e a pasamos a gris
parser = argparse.ArgumentParser(description='Detctor de esquinas de Harris.')
parser.add_argument('--input', help='Path a imaxe de entrada.', default='../data/chess.png')
args = parser.parse_args()
src = cv.imread(cv.samples.findFile(args.input))
if src is None:
    print('Non atopo a imaxe de entreda:', args.input)
    exit(0)
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
# Creamos un xanela e un trackbar
cv.namedWindow(source_window)
thresh = 200 # Limair inicial
cv.createTrackbar('Limiar: ', source_window, thresh, max_thresh, cornerHarris_demo)
cv.imshow(source_window, src)
cornerHarris_demo(thresh)
cv.waitKey()
