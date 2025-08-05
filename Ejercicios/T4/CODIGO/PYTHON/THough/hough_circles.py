import cv2
import numpy as np
import sys

def onTrackbarChange(max_slider):
    cimg = np.copy(img)

    p1 = max_slider
    p2 = max_slider * 0.4

  # Detectamos circulos empregando a transforamda de HoughCircles
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, cimg.shape[0]/64, param1=p1, param2=p2, minRadius=25, maxRadius=50)

    # Se atopamos polo menos 1 circulo
    if circles is not None:
        cir_len = circles.shape[1] 
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # debuxamo o circulo exterior
            cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Debuxamos o centro
            cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
    else:
        cir_len = 0 # non se atoparon circulos
    
    # Visualizamos a imaxe de saida
    cv2.imshow('Imaxe', cimg)    

    # Imaxe de borde a efecto de depuracion
    edges = cv2.Canny(gray, p1, p2)
    cv2.imshow('Bordes', edges)

    

    
if __name__ == "__main__":
    # Lemos a imaxe
    img = cv2.imread(sys.argv[1], 1)
    if img is None:
        print("imaxe de entrada non atopada")

    # Convertemos a iamxe de gros
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Creamoa as xanelas de visualziacion
    cv2.namedWindow("Bordes")
    cv2.namedWindow("Imaxe")
    

    # Trackbar para cambiar o limiar
    initThresh = 105 
    maxThresh = 200 

    # Creamos o trackbar
    cv2.createTrackbar("Limiar", "Imaxe", initThresh, maxThresh, onTrackbarChange)
    onTrackbarChange(initThresh)
    
    while True:
        key = cv2.waitKey(1)
        if key == 27:   #tecla Esc
            break

    cv2.destroyAllWindows()
