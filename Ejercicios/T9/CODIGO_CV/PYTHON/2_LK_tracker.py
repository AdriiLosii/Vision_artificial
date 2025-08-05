import numpy as np
import cv2

videoFileName ="../data/race_car.mp4"

cap = cv2.VideoCapture(videoFileName)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('sparse-output.avi',cv2.VideoWriter_fourcc(*'MP4V'), 20, (width,height))


# parametros para o detector de  ShiTomasi
numCorners = 100
feature_params = dict( maxCorners = numCorners,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )


# Parametros para o detector de fluxo optico de lucas kanade
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# criamos cores aleatoriios
color = np.random.randint(0,255,(numCorners,3))

# Colle o primeiro frame e detectamos as esquinas nel
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
old_points = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Creamos unha imaxe mascara para debuxar os tracks
mask = np.zeros_like(old_frame)
count = 0
while(1):
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    count += 1
    # achamos o fluxo optico
    new_points, status, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_points, None, **lk_params)

    # Seleccionamos o mellores puntos
    good_new = new_points[status==1]
    good_old = old_points[status==1]

    # debuxamos os  tracks
    for i,(new,old) in enumerate(zip(good_new, good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        cv2.line(mask, (round(a),round(b)),(round(c),round(d)), color[i].tolist(), 2, cv2.LINE_AA)
        cv2.circle(frame,(round(a),round(b)),3,color[i].tolist(), -1)

    # visualizamos cada 5 frames
    display_frame = cv2.add(frame,mask)
    out.write(display_frame)
    cv2.imshow("Demo de fluxo optico LK", display_frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # agora actualiamos o frame anterior e os puntos anteriores
    old_gray = frame_gray.copy()
    old_points = good_new.reshape(-1,1,2)

cap.release()
out.release()
