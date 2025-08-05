import cv2
import numpy as np

filename = "../../data/images/sample.jpg"
image = cv2.imread(filename)
# Comprobamos se se leu a imaxe
if image is None:
    print("Non poiden ler a imaxe")

if image is None:  # Comprobamos se poidemos ler a imaxe
    print("Non poiden ler a imaxe")

kernel_size = 5
# Kernel de 5x5 con todos os elementos a 1
kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / kernel_size**2

# Visualizamos o kernel
print (kernel)

#Realizamos a convolucion
result = cv2.filter2D(image, -1, kernel, (-1, -1), delta=0, borderType=cv2.BORDER_DEFAULT)

cv2.imshow("Imaxe Orixinal", image)
cv2.imshow("Resultado da convolucion", result)
cv2.waitKey(0)

