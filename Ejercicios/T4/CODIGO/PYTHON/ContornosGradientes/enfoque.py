import cv2
import numpy as np

filename = "../../data/Toyota.jpg"

image = cv2.imread(filename)
if image is None:
    print("Non poiden ler a imaxe")


# Kernel de enfoque
sharpen = np.array((
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]), dtype="int")

# Empregamos a funcion filter2D para achar a convolucion co filtro de enfoque
sharpenOutput = cv2.filter2D(image, -1, sharpen)

cv2.imshow("Imaxe orixinal", image)
cv2.imshow("Resultado do enfoque", sharpenOutput)
cv2.waitKey(0)

