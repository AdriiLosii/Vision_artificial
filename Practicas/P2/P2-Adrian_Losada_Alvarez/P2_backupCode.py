import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")


# Función para obtener las coordenadas (x,y) donde se hace click con el ratón
pts = []
def pick_points(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pts.append((x,y))

# Función para separar en listas distintas los valores de las coordenadas 'x' e 'y' de las lineas dadas
def separate_x_y(lines):
    lines_x = []
    lines_y = []
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                lines_x.extend([x1, x2])
                lines_y.extend([y1, y2])

    return lines_x, lines_y

# Función para dibujar lineas en la imagen dada
def draw_lines(img, lines, color = [0, 0, 255], thickness = 4):
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)


# MAIN():
cap = cv2.VideoCapture('./PracticaXanelaCV/DATA/proba.mp4')
if not cap.isOpened():
    print("Error al abrir el video.")
    exit()

ret, image = cap.read()
if(ret):
    cv2.imshow('Primer fotograma', image)
    while(len(pts) < 4):
        cv2.setMouseCallback('Primer fotograma', pick_points)
        # Mostramos los puntos seleccionados
        if(len(pts) > 0):
            cv2.circle(image, pts[len(pts)-1], 5, (0,0,255), -1)
            cv2.imshow('Primer fotograma', image)

        if (cv2.waitKey(30) == ord('q')):
            break

    cv2.destroyAllWindows()

while(ret):
    ret, image = cap.read()
    if(not ret):
        break

    # Corregimos la distorsión de la ventana
    pts_src = np.array([pts[0], pts[3], pts[1], pts[2]], np.float32)
    pts_dst = np.array([(0,0), (0,image.shape[0]), (image.shape[1],0), (image.shape[1],image.shape[0])], np.float32)
    matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
    trans_image = cv2.warpPerspective(image, matrix, (image.shape[1],image.shape[0]))

    # Recortamos la imagen
    cropped_image = trans_image[0:trans_image.shape[0], trans_image.shape[1]//3:trans_image.shape[1]//3*2]
    
    # Aplicamos filtros:
    gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    mean_gray = np.mean(gray_image)

    # Obtenemos el valor de la media de grises de la imagen para determinar si es de dia o de noche
    if(mean_gray>=150):
        # Filtros diurnos
        blurred_image = cv2.GaussianBlur(gray_image, ksize=(31, 1), sigmaX=0)                       # Emborronamos la imagen con un kernel orientado en el eje X
        threshold = cv2.inRange(blurred_image, lowerb=90, upperb=205)                               # Filtramos los valores de grises
        cannyed_image = cv2.Canny(threshold, threshold1=350, threshold2=220)                        # Obtenemos los bordes
        dilated_image = cv2.dilate(cannyed_image, kernel=np.ones((3,3), np.uint8), iterations=1)    # Dilatamos las lineas de los bordes obtenidos para unificar lineas horizontales

        # Creamos las lineas de la transformada de Hough
        hough_lines = cv2.HoughLinesP(dilated_image, rho = 1, theta=np.pi/180, threshold = 60, minLineLength = 150, maxLineGap = 50)
    else:
        # Filtros nocturnos
        blurred_image = cv2.GaussianBlur(gray_image, ksize=(71, 1), sigmaX=0)                       # Emborronamos la imagen con un kernel orientado en el eje X
        threshold = cv2.inRange(blurred_image, lowerb=70, upperb=110)                               # Filtramos los valores de grises
        eroded_image = cv2.erode(threshold, kernel=np.ones((1,30), np.uint8), iterations=3)         # Erosionamos en el eje X
        ret, mask = cv2.threshold(eroded_image, thresh=180, maxval=255, type=cv2.THRESH_BINARY)     # Conversion a binario
        cannyed_image = cv2.Canny(mask, threshold1=350, threshold2=220)                             # Obtenemos los bordes
        dilated_image = cv2.dilate(cannyed_image, kernel=np.ones((3,3), np.uint8), iterations=1)    # Dilatamos las lineas de los bordes obtenidos para unificar lineas horizontales
        blurred_image2 = cv2.GaussianBlur(dilated_image, ksize=(31, 1), sigmaX=0)                   # Emborronamos la imagen una vez más para reducir el ruido

        # Creamos las lineas de la transformada de Hough
        hough_lines = cv2.HoughLinesP(blurred_image2, rho = 1, theta=np.pi/180, threshold = 60, minLineLength = 150, maxLineGap = 50)

    # Juntamos todas las lineas horizontales en una única para cada borde de la barra (una horizontal para superior, otra horizontal para el inferior)
    lines_x, lines_y = separate_x_y(hough_lines)

    top_y = []
    bottom_y = []
    for val in lines_y:
        if(min(lines_y) <= val <= min(lines_y)+15):
            # Valores de la horizontal superior
            top_y.append(val)
        else:
            # Valores de la horizontal inferior
            bottom_y.append(val)

    if(len(top_y)!=0): line_y_top = round(np.mean(top_y))
    else: line_y_top = 0
    if(len(bottom_y)!=0): line_y_bottom = round(np.mean(bottom_y))
    else: line_y_bottom = 0

    # Definimos las lineas y las agregamos a la imagen inicial
    min_x = 0
    max_x = trans_image.shape[1]
    min_y = 0
    max_y = trans_image.shape[0]

    square_top = [[[min_x, 0, max_x, 0], [max_x, 0, max_x, line_y_top], [min_x, line_y_top, max_x, line_y_top], [min_x, 0, min_x, line_y_top]]]
    square_bottom = [[[min_x, line_y_bottom, max_x, line_y_bottom], [max_x, line_y_bottom, max_x, max_y], [min_x, max_y, max_x, max_y], [min_x, line_y_bottom, min_x, max_y]]]

    lines_image = np.zeros((max_y, max_x, 3), dtype = np.uint8)
    if(line_y_top!=0):
        draw_lines(lines_image, square_top, color=[255,0,0])
    if(line_y_bottom!=0):
        draw_lines(lines_image, square_bottom, color=[0,0,255])
    final_image = cv2.addWeighted(trans_image, alpha = 0.7, src2=lines_image, beta = 1.0, gamma = 0.0)

    # Calculamos el porcentaje de apertura y cierre de la cortina
    if(line_y_top!=0 and line_y_bottom!=0):
        line_center = (line_y_top+line_y_bottom)/2
        percent_top = round(line_center/max_y*100, 2)
        percent_bottom = round(100-percent_top, 2)
    elif(line_y_top!=0 and line_y_bottom==0):
        percent_top = 100
        percent_bottom = 0
    elif(line_y_top==0 and line_y_bottom!=0):
        percent_top = 0
        percent_bottom = 100
    else:
        percent_top = None
        percent_bottom = None

    # Agregamos texto a la imagen
    cv2.putText(final_image, 'Cerrada: '+str(percent_top)+'%', org=(5, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=[255, 0, 0], thickness=2, lineType=1, bottomLeftOrigin=False)
    cv2.putText(final_image, 'Abierta: '+str(percent_bottom)+'%', org=(5, 60), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=[0, 0, 255], thickness=2, lineType=1, bottomLeftOrigin=False)

    # Visualizacion
    cv2.imshow('Imagen Final', final_image)

    if cv2.waitKey(1) == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
