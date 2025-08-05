import cv2
import numpy as np

# Lemos a imaxe
filename = "../../data/images/dark-flowers.jpg"
im = cv2.imread(filename)
# Comprobamos se se leu a imaxe
if im is None:
    print("Non poiden ler a imaxe")

imEq = np.zeros_like(im)

# Realizamos ecualizacion separadamente das canles
for i in range(0,3):
    imEq[:,:,i] = cv2.equalizeHist(im[:,:,i])

cv2.imshow("Imaxe Orixinal",im)
cv2.waitKey(0)
cv2.imshow("Histograma ecualizada",imEq)
cv2.waitKey(0)

# Lemos a imaxe de cor
filename = "../../data/images/dark-flowers.jpg"
im = cv2.imread(filename)
# Comprobamos se se leu a imaxe
if im is None:
    print("Non poiden ler a imaxe")

# Convertemos HSV
imhsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

# Ecualozamos so a candle V
imhsv[:,:,2] = cv2.equalizeHist(imhsv[:,:,2])

# Convertemos o formato BGR
imEq = cv2.cvtColor(imhsv, cv2.COLOR_HSV2BGR)

cv2.imshow("Imaxe Orixinal",im)
cv2.waitKey(0)
cv2.imshow("Histograma ecualizada",imEq)
cv2.waitKey(0)
