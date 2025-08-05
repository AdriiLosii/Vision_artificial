# Importacions estandar
import cv2
import numpy as np;

# Lemos a imaxe
im = cv2.imread("../data/blob_detection.jpg", cv2.IMREAD_GRAYSCALE)

# Detector cos parametros por defecto do detector.
detector = cv2.SimpleBlobDetector_create()

keypoints = detector.detect(im)

im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

# Marcamos os blobd
for k in keypoints:
    x,y = k.pt
    x=int(round(x))
    y=int(round(y))
    # Centro en negro
    cv2.circle(im,(x,y),5,(0,0,0),-1)
    # Radio do blob
    diameter = k.size
    radius = int(round(diameter/2))
    # O blob en vermello
    cv2.circle(im,(x,y),radius,(0,0,255),2)

# Visualizamos
cv2.imshow("Imaxe",im)
cv2.waitKey(0)

# Asignamos os parametros do SimpleBlobDetector.
params = cv2.SimpleBlobDetector_Params()

# Cambiomos os limiares
params.minThreshold = 10
params.maxThreshold = 200

# Filtramos por Area.
params.filterByArea = True
params.minArea = 1500

# Filtramos por circularidades
params.filterByCircularity = True
params.minCircularity = 0.1

# Filtramos por convexidade
params.filterByConvexity = True
params.minConvexity = 0.87

# Filtramos por Inercia
params.filterByInertia = True
params.minInertiaRatio = 0.01

# Creamos o detector cos parametros configurados
detector = cv2.SimpleBlobDetector_create(params)

####CONTINUA O CODIGO AQUI PARA APLICAR DE NOVO ESTE DETECTOR CONFIGURADO
