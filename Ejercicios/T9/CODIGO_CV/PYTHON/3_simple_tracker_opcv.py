#USA: python 3_simple_tracker_opcv.py 6
# selecciona u numero entre 0-7

import cv2
import sys

if __name__ == '__main__' :

# Configura o rastreador.
# Escolla un rastreador

    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'CSRT', 'MOSSE']
    tracker_type = tracker_types[int(sys.argv[1])]

    if tracker_type == 'BOOSTING':
        tracker = cv2.legacy.TrackerBoosting_create()
    elif tracker_type == 'MIL':
        tracker = cv2.legacy.TrackerMIL_create()
    elif tracker_type == 'KCF':
        tracker = cv2.legacy.TrackerKCF_create()
    elif tracker_type == 'TLD':
        tracker = cv2.legacy.TrackerTLD_create()
    elif tracker_type == 'MEDIANFLOW':
        tracker = cv2.legacy.TrackerMedianFlow_create()
    elif tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    elif tracker_type == "CSRT":
        tracker = cv2.legacy.TrackerCSRT_create()
    elif tracker_type == "MOSSE":
        tracker = cv2.legacy.TrackerMOSSE_create() #cv2.TrackerMOSSE_create()
    else:
        tracker = None
        print('Incorrect tracker name')
        print('Available trackers are:')
        for t in trackerTypes:
          print(t)


    # Ler o vídeo
    filename = "../data/cycle.mp4"
    cap = cv2.VideoCapture(filename)
    video_name = filename.split('/')[-1].split('.')[0]
    video = cv2.VideoCapture(filename)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Saír se o vídeo non está aberto.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    # Ler o primeiro cadro.
    ok, frame = video.read()
    if not ok:
        print('Non podemos atopar o video')
        sys.exit()

    # Define un cadro delimitador inicial
    # Cycle
    bbox = (477, 254, 55, 152)

    # ship
    # bbox = (751, 146, 51, 78)

    # Hockey
    # bbox = (129, 47, 74, 85)

    # Face2
    # bbox = (237, 145, 74, 88)

    # meeting
    # bbox = (627, 183, 208, 190)     #CSRT
    # bbox = (652, 187, 118, 123)       #KCF

    # surfing
    # bbox = (97, 329, 118, 293)

    # surf
    # bbox = (548, 587, 52, 87)

    # spinning
    # bbox = (232, 218, 377, 377)       #RED
    # bbox = (699, 208, 383, 391)         #BLUE

    # Car
    # bbox = (71, 457, 254, 188)

    # Descomenta a liña de abaixo para seleccionar unha caixa delimitadora diferente
    # bbox = cv2.selectROI(frame, False)
    print("obxectivo inicial fixado en : {}".format(bbox))
    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)
    out = cv2.VideoWriter('{}_{}_{}.mp4'.format(video_name,tracker_type,bbox),cv2.VideoWriter_fourcc(*'MP4V'), 30, (640,360))

    while True:
        # Ler un novo cadro
        ok, frame = video.read()
        if not ok:
            break

        # Horario de inicio
        timer = cv2.getTickCount()

        # Actualizar o rastreador
        ok, bbox = tracker.update(frame)

        # Calcular fotogramas por segundo (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # Debuxa un cadro delimitador
        if ok:
            # Seguimento do éxito
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (0,255,0), 2, 1)
        else :
            # Fallo de seguimento
            cv2.putText(frame, "Fallo de Tracking detectado", (20,120), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2, cv2.LINE_AA)

        # Mostra o tipo de rastreador no marco
        cv2.putText(frame, tracker_type + " Tracker", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0),2, cv2.LINE_AA);

        # Mostrar FPS no marco
        cv2.putText(frame, "FPS : " + str(int(fps)), (20,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA);


        # Mostrar resultado
        cv2.imshow("Tracking", frame)

        outframe = cv2.resize(frame, (640,360))
        out.write(outframe)

        # Saír se se preme ESC
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break
    out.release()
