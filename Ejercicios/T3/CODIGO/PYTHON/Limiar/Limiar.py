import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)
matplotlib.rcParams['image.cmap'] = 'gray'

# Lemos unha imaxe de gris
imagePath = '../../NOTEBOOKS/data/threshold.png'
src = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)

# Asignamos valores ao limiar e o Maxvalue
thresh = 100
maxValue = 255

def thresholdUsingLoop(src, thresh, maxValue):
    # Creamos a imaxe de saida
    dst = src.copy()
    height,width = src.shape[:2]

    # Lazo sobre filas
    for i in range(height):
        # Lazo sobre columnas
        for j in range(width):
            if src[i,j] > thresh:
                dst[i,j] = maxValue
            else:
                dst[i,j] = 0

    return dst

t = time.time()
dst = thresholdUsingLoop(src, thresh, maxValue)
print("Tempo = {} s".format(time.time() - t))

plt.figure(figsize=[15,15])
plt.subplot(121);plt.imshow(src);plt.title("Imaxe Orixinal");
plt.subplot(122);plt.imshow(dst);plt.title("Imaxe Limiar");
plt.show()

def thresholdUsingVectors(src, thresh, maxValue):
    # Creamos unha imaxe negra (todo ceros )
    dst = np.zeros_like(src)

    # Atopamos o pixels con valor>limiar
    thresholdedPixels = src>thresh

    # Asignamos estes pixel ao valor maxValue
    dst[thresholdedPixels] = maxValue

    return dst

t = time.time()
dst = thresholdUsingVectors(src, thresh, maxValue)
print("Tempo = {} s".format(time.time() - t))

plt.figure(figsize=[15,15])
plt.subplot(121);plt.imshow(src);plt.title("Imaxe Orixinal");
plt.subplot(122);plt.imshow(dst);plt.title("Imaxe Limiar");
plt.show()

t = time.time()
th, dst = cv2.threshold(src, thresh, maxValue, cv2.THRESH_BINARY)
print("Tempo = {} s".format(time.time() - t))

plt.figure(figsize=[15,15])
plt.subplot(121);plt.imshow(src);plt.title("Imaxe Orixinal");
plt.subplot(122);plt.imshow(dst);plt.title("Imaxe Limiar");
plt.show()

time_opencv = 0
time_loops = 0
time_vector = 0
n_samples = 10
for i in range(n_samples):

    t = time.time()
    dst = thresholdUsingLoop(src, thresh, maxValue)
    time_loops += time.time() - t

    t = time.time()
    dst = thresholdUsingVectors(src, thresh, maxValue)
    time_vector += time.time() - t

    t = time.time()
    th, dst = cv2.threshold(src, thresh, maxValue, cv2.THRESH_BINARY)
    time_opencv += time.time() - t

print("Tempo promedio empleado pola rutina de lazos = {} s".format(time_loops/n_samples))
print("Tempo promedio empleado pola rutina de vectoriazacion = {} s".format(time_vector/n_samples))
print("Tempo promedio empleado pola rutina de OpenCV = {} s".format(time_opencv/n_samples))

thresh = 100
maxValue = 150

th, dst_bin = cv2.threshold(src, thresh, maxValue, cv2.THRESH_BINARY)

th, dst_bin_inv = cv2.threshold(src, thresh, maxValue, cv2.THRESH_BINARY_INV)

th, dst_trunc = cv2.threshold(src, thresh, maxValue, cv2.THRESH_TRUNC)

th, dst_to_zero = cv2.threshold(src, thresh, maxValue, cv2.THRESH_TOZERO)

th, dst_to_zero_inv = cv2.threshold(src, thresh, maxValue, cv2.THRESH_TOZERO_INV)

print("Valor do limiar = {}, Max = {}".format(thresh, maxValue))
plt.figure(figsize=[20,12])
plt.subplot(231);plt.imshow(src, cmap='gray', vmin=0, vmax=255);plt.title("Orixinal");
plt.subplot(232);plt.imshow(dst_bin, cmap='gray', vmin=0, vmax=255);plt.title("Limiar Binario");
plt.subplot(233);plt.imshow(dst_bin_inv, cmap='gray', vmin=0, vmax=255);plt.title("Limiar Binario Inverso");
plt.subplot(234);plt.imshow(dst_trunc, cmap='gray', vmin=0, vmax=255);plt.title("Limiar truncado");
plt.subplot(235);plt.imshow(dst_to_zero, cmap='gray', vmin=0, vmax=255);plt.title("Limiar a cero");
plt.subplot(236);plt.imshow(dst_to_zero_inv, cmap='gray', vmin=0, vmax=255);plt.title("Limiar a cero invertido");


# Exemplo deteccion de liñas de estrada.
# Lemos a imaxe.
img = cv2.imread('../../NOTEBOOKS/data/road_lanes.png', cv2.IMREAD_GRAYSCALE)

# Limiar binaria.
retval, img_thresh = cv2.threshold(img, 165, 255, cv2.THRESH_BINARY)

# Visualizacion.
cv2.imshow('Imaxe', img)
cv2.imshow('Limiar', img_thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Lectuta de imaxe
img = cv2.imread('../../NOTEBOOKS/data/Piano_Sheet_Music.png', cv2.IMREAD_GRAYSCALE)

# Limar global.
retval, img_thresh_gbl_1 = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
retval, img_thresh_gbl_2 = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)

# Limiar adaptativo
img_thresh_adp = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, 7)

# Visualización.
cv2.imshow('Image', img)
cv2.imshow('Limiar global 1', img_thresh_gbl_1)
cv2.imshow('Limiar global 2', img_thresh_gbl_2)
cv2.imshow('Limiar adaptativo', img_thresh_adp)
cv2.waitKey(0)
cv2.destroyAllWindows()
