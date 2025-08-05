import sys
import cv2
from random import randint

trackerTypes = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

def createTrackerByName(trackerType):
  # Crea un rastreador baseado no nome do rastreador
  if trackerType == trackerTypes[0]:
    tracker = cv2.legacy.TrackerBoosting_create()
  elif trackerType == trackerTypes[1]:
    tracker = cv2.legacy.TrackerMIL_create()
  elif trackerType == trackerTypes[2]:
    tracker = cv2.legacy.TrackerKCF_create()
  elif trackerType == trackerTypes[3]:
    tracker = cv2.legacy.TrackerTLD_create()
  elif trackerType == trackerTypes[4]:
    tracker = cv2.legacy.TrackerMedianFlow_create()
  elif trackerType == trackerTypes[5]:
    tracker = cv2.legacy.TrackerGOTURN_create()
  elif trackerType == trackerTypes[6]:
    tracker = cv2.legacy.TrackerMOSSE_create()
  elif trackerType == trackerTypes[7]:
    tracker = cv2.legacy.TrackerCSRT_create()
  else:
    tracker = None
    print('Nome do tracker incorrecto')
    print('os disponhibles son:')
    for t in trackerTypes:
      print(t)

  return tracker

if __name__ == '__main__':

  for t in trackerTypes:
      print(t)

  trackerType = "CSRT"

  # Establece o vídeo para cargar
  filename = "../data/cycle.mp4"

  # Crea un obxecto de captura de vídeo para ler vídeos
  cap = cv2.VideoCapture(filename)
  video_name = filename.split('/')[-1].split('.')[0]
  out = cv2.VideoWriter('{}_{}.mp4'.format(video_name,trackerType),cv2.VideoWriter_fourcc(*'MP4V'), 30, (640,360))

  # Ler o primeiro cadro
  success, frame = cap.read()
  # saír se non pode ler o ficheiro de vídeo
  if not success:
    print('Non atopamos o video')
    sys.exit(1)

  ## deleccionamos as caixas
  bboxes = []
  ## selecionamos as caixas
  colors = []
  for i in range(3):
    # Selecciona algunhas cores aleatorias
    colors.append((randint(64, 255), randint(64, 255),
                randint(64, 255)))
  # Seleccione os cadros delimitadores
  bboxes = [(471, 250, 66, 159), (349, 232, 69, 102)]
  # print('Caixas seleccionadas {}'.format(bboxes))

  # Crear obxecto MultiTracker
  multiTracker = cv2.legacy.MultiTracker_create()

  # Inicializar MultiTracker
  for bbox in bboxes:
    multiTracker.add(createTrackerByName(trackerType), frame, bbox)


  # Procesar vídeo e rastrexar obxectos
  while cap.isOpened():
    success, frame = cap.read()
    if not success:
      break

    # obter a localización actualizada dos obxectos en cadros posteriores
    success, boxes = multiTracker.update(frame)

    #debuxa obxectos rastrexados
    for i, newbox in enumerate(boxes):
      p1 = (int(newbox[0]), int(newbox[1]))
      p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
      cv2.rectangle(frame, p1, p2, colors[i], 2, 1)

    #mostrar marco
    cv2.imshow('MultiTracker', frame)
    outframe = cv2.resize(frame, (640,360))
    out.write(outframe)


    # sair no botón ESC
    if cv2.waitKey(1) & 0xFF == 27:  # presiona Esc
      break
  out.release()
