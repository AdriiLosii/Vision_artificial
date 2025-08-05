import cv2
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)
matplotlib.rcParams['image.cmap'] = 'gray'

imageName = "../../data/opening.png"
# Imaxe de entrada
image = cv2.imread(imageName, cv2.IMREAD_GRAYSCALE)

# Check for invalid input
if image is None:
    print("Non atopo a imaxe")

cv2.imshow("imaxe",image)
cv2.waitKey(0)

# Especificamos o kernel
kernelSize = 10

# Creamos o kernel
element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*kernelSize+1, 2*kernelSize+1),
                                    (kernelSize, kernelSize))

# Erosion
imEroded = cv2.erode(image, element, iterations=1)
# Dilatacion
imOpen = cv2.dilate(imEroded, element, iterations=1)

# Visualizamos o kernel
cv2.imshow("imaxe",element*255)
cv2.waitKey(0)

# Visualizamos 
plt.figure(figsize=[15,15])
plt.subplot(131);plt.imshow(image);plt.title("Imaxe Orixinal")
plt.subplot(132);plt.imshow(imEroded,cmap='gray');plt.title("Despois da erosion")
plt.subplot(133);plt.imshow(imOpen,cmap='gray');plt.title("Despois da dilatacion")
plt.show()

# Kernel para a operacion de apertura
openingSize = 3

# forma eliptica do kernel
element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
            (2 * openingSize + 1, 2 * openingSize + 1),
            (openingSize,openingSize))

imageMorphOpened = cv2.morphologyEx(image, cv2.MORPH_OPEN,
                        element,iterations=3)
plt.figure(figsize=[15,15])
plt.subplot(121);plt.imshow(image);plt.title("Imaxe Orixianla")
plt.subplot(122);plt.imshow(imageMorphOpened);plt.title("Despois da apertura")
plt.show()

imageName = "../../data/closing.png"
# Imaxe para o peche
image = cv2.imread(imageName, cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Non atopo a imaxe")

# Tamanho do kernel
kernelSize = 10

# Creamos o kernel
element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*kernelSize+1, 2*kernelSize+1),
                                    (kernelSize, kernelSize))

# Dilatacion
imDilated = cv2.dilate(image, element)
# Erosion
imClose = cv2.erode(imDilated, element)

plt.figure(figsize=[15,15])
plt.subplot(131);plt.imshow(image);plt.title("Imaxe orixinal")
plt.subplot(132);plt.imshow(imDilated,cmap='gray');plt.title("Despois da dilatacion")
plt.subplot(133);plt.imshow(imClose,cmap='gray');plt.title("Despois da erosion")

plt.show()

# Kernel para operacion de peche
closingSize = 10

# Forma eliptica
element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
            (2 * closingSize + 1, 2 * closingSize + 1),
            (closingSize,closingSize))

imageMorphClosed = cv2.morphologyEx(image,
                                    cv2.MORPH_CLOSE, element)
plt.figure(figsize=[15,15])
plt.subplot(121);plt.imshow(image);plt.title("Imaxe orixianl")
plt.subplot(122);plt.imshow(imageMorphClosed,cmap='gray');plt.title("Despois do peche")
plt.show()

