import numpy as np
import cv2,sys,time

if __name__ == '__main__':

  filename =  "../data/face2.mp4"
  cap = cv2.VideoCapture(filename)

  # Lemos un frame e atopamos a cara empregando dlib
  ret,frame = cap.read()

  #  Detectamos a cara na imaxe
  faceCascade = cv2.CascadeClassifier('../data/models/haarcascade_frontalface_default.xml')

  frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  faces = faceCascade.detectMultiScale(frameGray,1.3,5)
  x,y,w,h = faces[0]

  currWindow = (x,y,w,h)
  #  conseguimos a rexion da cara no frame
  roiObject = frame[y:y+h,x:x+w]
  hsvObject =  cv2.cvtColor(roiObject, cv2.COLOR_BGR2HSV)

  # obtén a máscara para calcular o histograma do obxecto e tamén elimina o ruído
  mask = cv2.inRange(hsvObject, np.array((0., 50., 50.)), np.array((180.,255.,255.)))
  cv2.imshow("mascara",mask)
  cv2.imshow("Obxecto",roiObject)

  # Busca o histograma e normalízao para que teña valores entre 0 e 255
  histObject = cv2.calcHist([hsvObject], [0], mask, [180], [0,180])
  cv2.normalize(histObject, histObject, 0, 255, cv2.NORM_MINMAX)

  # Configura os criterios de terminación, xa sexa 10 iteracións ou móvese polo menos 1 punto
  term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
  i=0
  while(1):
    ret, frame = cap.read()

    if ret == True:
      # Converter a espazo de cor hsv
      hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

      # atopar a imaxe posterior proxectada co histograma obtido anteriormente
      backProjectImage = cv2.calcBackProject([hsv], [0], histObject, [0,180], 1)
      cv2.imshow("Imaxe reproxectada", backProjectImage)

      # Calcula a nova xanela usando CAM shift no marco actual
      rotatedWindow, currWindow = cv2.CamShift(backProjectImage, currWindow, term_crit)

      # Emprega a xanela polo cambio medio
      x,y,w,h = currWindow

      # Obter os vértices de rotatedWindow
      rotatedWindow = cv2.boxPoints(rotatedWindow)
      rotatedWindow = np.int0(rotatedWindow)
      frameClone = frame.copy()

      # Mostra a xanela actual utilizada para o cambio medio
      cv2.rectangle(frameClone, (x,y), (x+w,y+h), (255, 0, 0), 2, cv2.LINE_AA)

      # Mostra o rectángulo xirado coa información de orientación
      frameClone = cv2.polylines(frameClone, [rotatedWindow], True, (0, 255, 0), 2, cv2.LINE_AA)
      cv2.putText(frameClone, "{},{},{},{}".format(x,y,w,h), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 255), 2, cv2.LINE_AA)
      cv2.putText(frameClone, "{}".format(rotatedWindow), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 255), 2, cv2.LINE_AA)
      cv2.imshow('tracker CAM Shift',frameClone)
      # cv2.imwrite('{:03d}.jpg'.format(i),frameClone)
      i+=1

      k = cv2.waitKey(10) & 0xff
      if k == 27:
        break
    else:
      break

  cv2.destroyAllWindows()
  cap.release()
