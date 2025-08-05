import cv2

# Usa source = 0 para especificar a webcam como fonte de video, OU
# especifica a ruta e nome ao ficheiro de video a ler.
#source = '../../data/videos/chaplin.mp4'
source = 0 #web-cam
# Creamos un obxecto para capturar o video da clase VideoCapture.
video_cap = cv2.VideoCapture(source)

#Lemos as caracteristicas do video e as imprimimos
assert video_cap.isOpened(), "conecta a webcam"
w   = video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h   = video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = video_cap.get(cv2.CAP_PROP_FPS)
print(f'{w:.0f}x{h:.0f} {fps}fps')




# Creamos unha xanela para visualizar o video.
win_name = 'Video'
cv2.namedWindow(win_name)

# Lazo para ler e visualizar un frame cada instante.
while True:
    # Lemos un frame co obxecto videoCapture.
    has_frame, frame = video_cap.read()
    if not has_frame:
        break
    # Visualizamos o frame
    cv2.imshow(win_name, frame)

    # Usa waitKey() para parar a execucion e dar tempo a visualizacion.
    # key = cv2.waitKey(0) espera indefinadamente ata que presiones unha tecla.
    # key = cv2.waitKey(1) espera 1 ms
    key = cv2.waitKey(6)

    #  waitKey() devolve o valor da tecla presionada.
    # Se o usuario selecciona `q` saimos do lazo do video stream.
    if key == ord('Q') or key == ord('q') or key == 27:
        # rompemos o lazo.
        break

video_cap.release()
cv2.destroyWindow(win_name)
