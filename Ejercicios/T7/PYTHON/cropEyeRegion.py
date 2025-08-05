import cv2
import numpy as np

#cargamos o haar-cascade entrenado para detectar caras frontais
face_cascade = cv2.CascadeClassifier('../data/models/haarcascade_frontalface_default.xml')

def getCroppedEyeRegion(targetImage):
    
    targetImageGray = cv2.cvtColor(targetImage, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(targetImageGray,1.3,5)
    x,y,w,h = faces[0]

    face_roi = targetImage[y:y+h,x:x+w]
    face_height,face_width = face_roi.shape[:2]

    # Aplicamos a formula heuristica para atopar
    # a rexion dos ollos unha vez atopada a rexion da cara
    eyeTop = int(1/6.0*face_height)
    eyeBottom = int(3/6.0*face_height)
    print("Altura dos ollos entre : {},{}".format(eyeTop,eyeBottom))
    
    eye_roi = face_roi[eyeTop:eyeBottom,:]
    
    # Redimensionamos o tamanho dos ollos ao fixado na base de datos de 96x32
    cropped = cv2.resize(eye_roi,(96, 32), interpolation = cv2.INTER_CUBIC)
    
    return cropped
