# Importamos paquetes necesarios
import cv2
import numpy as np


# Lemos a imaxe
image = cv2.imread("../../data/images/boy.jpg")

brightnessOffset = 50

# offset para incrementar o brilo
brightHighOpenCV = cv2.add(image, np.ones(image.shape,dtype='uint8')*brightnessOffset)

brightHighInt32 = np.int32(image) + brightnessOffset
brightHighInt32Clipped = np.clip(brightHighInt32,0,255)

# Visualizamos os resultados
cv2.imshow("Imaxe orixinal",image)
cv2.imshow("empregando a funcion cv2.add",brightHighOpenCV)
cv2.imshow("Usando numpy e recortando", brightHighInt32Clipped)
cv2.waitKey(0)
cv2.imwrite("../../data/images/brightHighOpenCV.png",brightHighOpenCV)
cv2.imwrite("../../data/images/brightHighInt32Clipped.png",brightHighInt32Clipped)

# Sumamos un offset para incrementar o brilo
brightHighFloat32 = np.float32(image) + brightnessOffset
brightHighFloat32NormalizedClipped = np.clip(brightHighFloat32/255,0,1)

brightHighFloat32ClippedUint8 = np.uint8(brightHighFloat32NormalizedClipped*255)

# Display the outputs
cv2.imshow("Imaxe orixinal",image)
cv2.imshow("Empregando np.float32 e recortando",brightHighFloat32NormalizedClipped)
cv2.imshow("Usando int->float->int e recortamos", brightHighFloat32ClippedUint8)
cv2.waitKey(0)
cv2.imwrite("../../data/images/brightHighFloat32NormalizedClipped.png",brightHighFloat32NormalizedClipped)
cv2.imwrite("../../data/images/brightHighFloat32ClippedUint8.png",brightHighFloat32ClippedUint8)
