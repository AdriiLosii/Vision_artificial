import cv2
import numpy as np
import sys

def onTrackbarChange(max_slider):
	global img
	global dst
	global gray

	dst = np.copy(img)

	th1 = max_slider 
	th2 = th1 * 0.4
	edges = cv2.Canny(img, th1, th2)
	
	# Tranformada de Houg probabilistica para rectas
	lines = cv2.HoughLinesP(edges, 2, np.pi/180.0, 50, minLineLength=10, maxLineGap=100)

	# Debuxamos as linhas
	for line in lines:
		x1, y1, x2, y2 = line[0]
		cv2.line(dst, (x1, y1), (x2, y2), (0,0,255), 1)

	cv2.imshow("Imaxe Resultado", dst)	
	cv2.imshow("Bordes",edges)

if __name__ == "__main__":
	
	# Lemos a imaxe
	img = cv2.imread('./data/lanes.jpg')
	if img is None:
		print("Non atopo a imaxe")
	
	# Copia para usos posteriores
	dst = np.copy(img)

	# Convertemos a gros
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Creamos xanelas de bisualizacion
	cv2.namedWindow("Bordes")
	cv2.namedWindow("Imaxe Resultado")
	  

	# Inicializamos o limiar
	initThresh = 500

	# Maximo valor do limiar
	maxThresh = 1000

	cv2.createTrackbar("Limiar", "Imaxe Resultado", initThresh, maxThresh, onTrackbarChange)
	onTrackbarChange(initThresh)

	while True:
		key = cv2.waitKey(1)
		if key == 27:   #Pulsa Esc
			break

	cv2.destroyAllWindows()
