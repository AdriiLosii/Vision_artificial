# EMPREGA
# python espazos_cor.py --image ../data/nome_imaxe.jpg

# importacion de paquetes
import argparse
import cv2

# Analizamos os argumentos de entrada
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, help="path a imaxe de entrda")
args = vars(ap.parse_args())

# cargamos a imaxe orixnal e a visualizamos
image = cv2.imread(args["image"])
cv2.imshow("RGB", image)

# lazo sobre cada canle da imaxe e a visualizamos
for (name, chan) in zip(("B", "G", "R"), cv2.split(image)):
	cv2.imshow(name, chan)

# esperamos a que pulsemos unha tecla (sobre a xanela de visualizacion!), 
# e despois pechamos todas as ventas
cv2.waitKey(0)
cv2.destroyAllWindows()

# convertemos a imaxe ao espazo HSV e visualizamos
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow("HSV", hsv)

# lazo sobre cada canle da imaxe e a visualizamos
for (name, chan) in zip(("H", "S", "V"), cv2.split(hsv)):
	cv2.imshow(name, chan)

# esperamos a que pulsemos unha tecla (sobre a xanela de visualizacion!), 
# e despois pechamos todas as ventas
cv2.waitKey(0)
cv2.destroyAllWindows()

# convertemos a imaxe ao espazo L*a*b* e visualizamos
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
cv2.imshow("L*a*b*", lab)

# lazo sobre cada canle da imaxe e a visualizamos
for (name, chan) in zip(("L*", "a*", "b*"), cv2.split(lab)):
	cv2.imshow(name, chan)

# esperamos a que pulsemos unha tecla (sobre a xanela de visualizacion!), 
# e despois pechamos todas as ventas
cv2.waitKey(0)
cv2.destroyAllWindows()

# visualizamos a imaxe orixinal e a version en gris
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Orixinal", image)
cv2.imshow("EscalaGris", gray)
cv2.waitKey(0)