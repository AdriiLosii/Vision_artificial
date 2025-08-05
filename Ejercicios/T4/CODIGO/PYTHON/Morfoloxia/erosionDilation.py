import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)
matplotlib.rcParams['image.cmap'] = 'gray'

imageName = "../../data/dilation_example.jpg"

# Lemos a imaxe de entrada
image = cv2.imread(imageName)

# Comprobamos o exito da lectura
if image is None:
    print("Non atopo a imaxe")
cv2.imshow("imaxe",image)
cv2.waitKey(0)

# Kernel para a dilatacion
kSize = (7,7)
kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kSize)
cv2.imshow("imaxe",kernel1*255)
cv2.waitKey(0)

# Dilatamos
imageDilated = cv2.dilate(image, kernel1)

plt.figure(figsize=[15,15])
plt.subplot(121);plt.imshow(image);plt.title("Imaxe oridinal")
plt.subplot(122);plt.imshow(imageDilated);plt.title("Imaxe dilatada")
plt.show()

# elemento estrutural para a dilatacion
kSize = (3,3)
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kSize)
cv2.imshow("imaxe",255*kernel2)
cv2.waitKey(0)

# dilatamos a imaxe de entrada
imageDilated1 = cv2.dilate(image, kernel2, iterations=1)
imageDilated2 = cv2.dilate(image, kernel2, iterations=2)

plt.figure(figsize=[20,20])
plt.subplot(131);plt.imshow(image);plt.title("Imaxe Orixinal")
plt.subplot(132);plt.imshow(imageDilated1);plt.title("Dilatacion iter 1")
plt.subplot(133);plt.imshow(imageDilated2);plt.title("Dilatacion iter  2")
plt.show()

# Imaxe de entrada
imageName = "../../data/erosion_example.jpg"
image = cv2.imread(imageName, cv2.IMREAD_COLOR)
# lemos
if image is None:
    print("Non atopo a imaxe")
cv2.imshow("imaxe",image)
cv2.waitKey(0)

# Erosionar a imaxe decrementa o seu brilo
imageEroded = cv2.erode(image, kernel1)

plt.figure(figsize=[15,15])
plt.subplot(121);plt.imshow(image);plt.title("Imaxe orixinal")
plt.subplot(122);plt.imshow(imageEroded);plt.title("Imaxe erosionada")
plt.show()


