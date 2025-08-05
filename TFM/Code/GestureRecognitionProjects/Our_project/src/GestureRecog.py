import cv2
import numpy as np
import pickle
from skimage.feature import hog
from skimage.filters import sobel
from imutils import auto_canny


# Variables globales para el histograma de la mano y puntos de recorrido
hand_hist = None
traverse_point = []
total_rectangle = 9
hand_rect_one_x = None
hand_rect_one_y = None
hand_rect_two_x = None
hand_rect_two_y = None


def process_image(hand_img):
    # Image processing
    if hand_img is None or hand_img.ndim != 3:
        raise ValueError("Archivo de imagen no válido o no en formato BGR.")
    
    # Convertir a escala de grises
    img_gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
    
    # Redimensionar la imagen
    img_resized = cv2.resize(img_gray, (128, 128))

    # Aplicar un filtro de Sobel para resaltar bordes
    img_sobel = sobel(img_resized)

    # Extraer las características HOG de la imagen filtrada
    hog_features, _  = hog(img_sobel, visualize=True)

    return hog_features


def draw_square(frame, centroid, square_size=400):
    """
    Dibuja un cuadrado alrededor del centroide de la mano.

    Parameters:
    - frame: Imagen de entrada.
    - centroid: Coordenadas del centroide de la mano.
    - square_size: Tamaño del cuadrado en píxeles.
    """
    center_x, center_y = centroid

    # Calcular las esquinas del cuadrado
    square_half_size = square_size // 2
    square_top_left = (max(center_x - square_half_size, 0), max(center_y - square_half_size, 0))
    square_bottom_right = (min(center_x + square_half_size, frame.shape[1]), min(center_y + square_half_size, frame.shape[0]))

    # Dibujar el cuadrado
    cv2.rectangle(frame, square_top_left, square_bottom_right, (255, 0, 0), 2)

def draw_prediction(frame, centroid, pred, square_size=400):
    """
    Dibuja la predicción sobre la detección de la mano.

    Parameters:
    - frame: Imagen de entrada.
    - centroid: Coordenadas del centroide de la mano.
    - square_size: Tamaño del cuadrado en píxeles.
    """
    center_x, center_y = centroid

    # Calcular las esquinas del cuadrado
    square_half_size = square_size // 2
    square_top_left = (max(center_x - square_half_size, 0), max(center_y - square_half_size, 0))
    square_top_right = (square_top_left[0] + square_size, square_top_left[1])

    # Dibujamos el fondo para mejor visualización
    offset = 90
    cv2.rectangle(frame, (square_top_left[0], square_top_left[1] - offset), (square_top_right[0], square_top_right[1]), (255, 0, 0), cv2.FILLED)

    # Dibujamos la predicción
    offset = 7
    cv2.putText(frame, str(pred), (square_top_left[0] + offset, square_top_left[1] - offset), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5, cv2.LINE_AA)

def crop_frame(frame, centroid, square_size=400):
    """
    Recorta el frame para mostrar solo la región dentro del cuadrado alrededor del centroide de la mano.

    Parameters:
    - frame: Imagen de entrada.
    - centroid: Coordenadas del centroide de la mano.
    - square_size: Tamaño del cuadrado en píxeles.

    Returns:
    - Región recortada del frame.
    """
    center_x, center_y = centroid

    # Calcular las esquinas del cuadrado
    square_half_size = square_size // 2
    square_top_left = (max(center_x - square_half_size, 0), max(center_y - square_half_size, 0))
    square_bottom_right = (min(center_x + square_half_size, frame.shape[1]), min(center_y + square_half_size, frame.shape[0]))

    # Recortar la región del frame
    cropped_frame = frame[square_top_left[1]:square_bottom_right[1], square_top_left[0]:square_bottom_right[0]]

    return cropped_frame

def contours(hist_mask_image):
    """
    Encuentra y devuelve contornos en una imagen.

    Parameters:
    - hist_mask_image: Imagen de entrada.

    Returns:
    - Lista de contornos.
    """
    gray_hist_mask_image = cv2.cvtColor(hist_mask_image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_hist_mask_image, 0, 255, 0)
    cont, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return cont

def draw_rect(frame):
    """
    Dibuja rectángulos en una imagen.

    Parameters:
    - frame: Imagen de entrada.

    Returns:
    - Imagen con rectángulos dibujados.
    """
    rows, cols, _ = frame.shape
    global total_rectangle, hand_rect_one_x, hand_rect_one_y, hand_rect_two_x, hand_rect_two_y

    # Primero se crea un único rectángulo
    hand_rect_one_x = np.array([6 * rows / 20, 6 * rows / 20, 6 * rows / 20, 9 * rows / 20, 9 * rows / 20, 9 * rows / 20, 12 * rows / 20, 12 * rows / 20, 12 * rows / 20], dtype=np.uint32)
    hand_rect_one_y = np.array([9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20, 10 * cols / 20, 11 * cols / 20], dtype=np.uint32)
    hand_rect_two_x = hand_rect_one_x + 10
    hand_rect_two_y = hand_rect_one_y + 10

    # Posteriormente se crea la "malla" de rectángulos
    for i in range(total_rectangle):
        cv2.rectangle(frame, (hand_rect_one_y[i], hand_rect_one_x[i]), (hand_rect_two_y[i], hand_rect_two_x[i]), (0, 255, 0), 1)

    return frame

def hand_histogram(frame):
    """
    Calcula y normaliza el histograma de la mano en una región de interés (ROI) de la imagen.

    Parameters:
    - frame: Imagen de entrada.

    Returns:
    - Histograma de la mano normalizado.
    """
    global hand_rect_one_x, hand_rect_one_y

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi = np.zeros([90, 10, 3], dtype=hsv_frame.dtype)

    for i in range(total_rectangle):
        roi[i * 10: i * 10 + 10, 0: 10] = hsv_frame[hand_rect_one_x[i]:hand_rect_one_x[i] + 10, hand_rect_one_y[i]:hand_rect_one_y[i] + 10]

    hand_hist = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])

    return cv2.normalize(hand_hist, hand_hist, 0, 255, cv2.NORM_MINMAX)

def hist_masking(frame, hist):
    """
    Crea una máscara basada en el histograma para resaltar la mano en la imagen.

    Parameters:
    - frame: Imagen de entrada.
    - hist: Histograma de la mano.

    Returns:
    - Imagen con la máscara aplicada.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)

    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    cv2.filter2D(dst, -1, disc, dst)

    ret, thresh = cv2.threshold(dst, 150, 255, cv2.THRESH_BINARY)

    thresh = cv2.merge((thresh, thresh, thresh))

    return cv2.bitwise_and(frame, thresh)

def centroid(max_contour):
    """
    Calcula el centroide de un contorno.

    Parameters:
    - max_contour: Contorno de entrada.

    Returns:
    - Coordenadas (cx, cy) del centroide.
    """
    moment = cv2.moments(max_contour)
    if moment['m00'] != 0:
        cx = int(moment['m10'] / moment['m00'])
        cy = int(moment['m01'] / moment['m00'])
        return cx, cy
    else:
        return None

def manage_image_opr(frame, hand_hist):
    """
    Realiza operaciones de procesamiento de imagen basadas en el histograma de la mano.

    Parameters:
    - frame: Imagen de entrada.
    - hand_hist: Histograma de la mano.
    """
    hist_mask_image = hist_masking(frame, hand_hist)
    hist_mask_image = cv2.erode(hist_mask_image, None, iterations=2)
    hist_mask_image = cv2.dilate(hist_mask_image, None, iterations=2)

    contour_list = contours(hist_mask_image)
    max_cont = max(contour_list, key=cv2.contourArea)

    cnt_centroid = centroid(max_cont)
    cv2.circle(frame, cnt_centroid, 5, [255, 0, 255], -1)

    if cnt_centroid is not None:
        draw_square(frame, cnt_centroid, square_size=400)
        cropped_frame = crop_frame(frame, cnt_centroid, square_size=400)

        return cropped_frame, cnt_centroid


def main():
    global hand_hist
    is_hand_hist_created = False
    capture = cv2.VideoCapture(0)

    # load model
    with open('model.pkl', 'rb') as f:
        svm = pickle.load(f)

    while capture.isOpened():
        pressed_key = cv2.waitKey(1)
        _, frame = capture.read()
        frame = cv2.flip(frame, 1)

        if pressed_key & 0xFF == ord('z'):
            is_hand_hist_created = True
            hand_hist = hand_histogram(frame)

        if is_hand_hist_created:
            # Obtenemos el frame de la mano
            cropped_frame, centroid = manage_image_opr(frame, hand_hist)

            # Procesamos la imagen
            data = process_image(cropped_frame)

            # Reformatear los datos para tener una forma 2D (1, n_features)
            data_reshaped = data.reshape(1, -1)

            # Obtenemos la predicción
            pred = svm.predict(data_reshaped)

            # Mostramos la predicción en terminal y en la imagen
            print(pred)
            draw_prediction(frame, centroid, pred[0], square_size=400)

        else:
            draw_rect(frame)

        cv2.imshow("Cámara", frame)

        if pressed_key == 27:
            break

    cv2.destroyAllWindows()
    capture.release()

if __name__ == '__main__':
    main()