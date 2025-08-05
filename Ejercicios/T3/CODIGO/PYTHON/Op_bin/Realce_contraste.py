# Importamos paquetes necesarios
import cv2
import numpy as np


# Lemos a imaxe
image = cv2.imread("../../data/images/boy.jpg")

contrastPercentage = 30

# Multiplicamos por un factor para incrementar o contraste
contrastHigh = image * (1+contrastPercentage/100)

# Visualizamos as saidas
cv2.imshow("Original Image",image)
cv2.imshow("High Contrast",contrastHigh)
cv2.waitKey(0)
cv2.imwrite("../../data/images/highContrast.png",contrastHigh)

print("Tipo de datos da Imaxe Orixinal : {}".format(image.dtype))
print("Tipo de datos da Imaxe alto contraste : {}".format(contrastHigh.dtype))

print("Maxima intensidade da imaxe orixinal : {}".format(image.max()))
print("Maxima intensidade da imaxe de alto contraste : {}".format(contrastHigh.max()))

#quremos unha imaxe contrasta nun 30% superior a orixinal
contrastPercentage = 30

# Recortamos os valores a [0,255] e rotornamos a tipo uint8 para visualzacion
contrastImage = image * (1+contrastPercentage/100)
clippedContrastImage = np.clip(contrastImage, 0, 255)
contrastHighClippedUint8 = np.uint8(clippedContrastImage)

# Convertemos ao rango [0,1] e mantemonos en formato float
contrastHighNormalized = (image * (1+contrastPercentage/100))/255
contrastHighNormalized01Clipped = np.clip(contrastHighNormalized,0,1)

cv2.imshow("Imaxe Orixinal",image)
cv2.imshow("convertida de novo a  uint8",contrastHighClippedUint8)
cv2.imshow("Normalizado a imaxe float a [0, 1]",contrastHighNormalized01Clipped)
cv2.waitKey(0)
cv2.imwrite("../../data/images/contrastHighClippedUint8.png",contrastHighClippedUint8)
cv2.imwrite("../../data/images/contrastHighNormalized01Clipped.png",contrastHighNormalized01Clipped)
