import cv2
import numpy as np

# Lemos o video
#input_video = 0
input_video = '../../data/videos/chaplin.mp4'
source = input_video  # source = 0 para webcam.

video_cap = cv2.VideoCapture(source)

if (video_cap.isOpened() == False):
	print("Error na apertura da webcam ou do ficheiro")

# Visualizamos o primeiro fframa.
ret, frame = video_cap.read()
cv2.imshow('Primeiro Frame', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Recuperamos as propieades do video.
frame_w   = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h   = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_fps = int(video_cap.get(cv2.CAP_PROP_FPS))

# Especificamos os valores para  fourcc.
fourcc_avi = cv2.VideoWriter_fourcc('M','J','P','G')
fourcc_mp4 = cv2.VideoWriter_fourcc(*'mp4v')

# Especificamos os nomes.
file_out_avi = 'video_out.avi'
file_out_mp4 = 'video_out.mp4'

# enlentecemos a velocidade do video.
frame_fps = int(frame_fps/3)

# Creamos os obxectos para escribir o video
out_avi = cv2.VideoWriter(file_out_avi, fourcc_avi, frame_fps, (frame_w,frame_h))
out_mp4 = cv2.VideoWriter(file_out_mp4, fourcc_mp4, frame_fps, (frame_w,frame_h))

# Funcion para anotar os frames.
def drawBannerText(frame, text, banner_height_percent = 0.05, text_color = (0,255,0)):
    # Debuxamos un banner negro na parte superior do frame.
    # porcentaxes: establecemos a altura do banner como un porcentaxe da altura do frame.
    banner_height = int(banner_height_percent * frame.shape[0])
    cv2.rectangle(frame, (0,0), (frame.shape[1],banner_height), (0,0,0), thickness = -1)

    # Debuxamos o banner.
    left_offset = 20
    location = (left_offset, int( 5 + (banner_height_percent * frame.shape[0])/2 ))
    fontScale = 1.5
    fontThickness = 2
    cv2.putText(frame, text, location, cv2.FONT_HERSHEY_PLAIN, fontScale, text_color,
        fontThickness, cv2.LINE_AA)

# Procesamos os frames do video
# Lemos todos os frames do video
frame_count = 0
while True:

    # Lemos un frame cada instante.
    ok, frame = video_cap.read()
    if not ok:
        break

    # Incrementamos o contados dos frames para a anotacion.
    frame_count += 1

    # Anotamos o frame do video.
    drawBannerText(frame, 'Frame: ' + str(int(frame_count)) + ' FPS: ' + str(int(frame_fps)))

    # Escribiemo o frame nos ficeiros de saida.
    out_avi.write(frame)
    out_mp4.write(frame)

# Liberamos memoria
video_cap.release()
out_avi.release()
out_mp4.release()
