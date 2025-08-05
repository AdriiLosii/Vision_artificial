import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['figure.figsize'] = (10.0, 10.0)
matplotlib.rcParams['image.cmap'] = 'gray'

# Cargamos la imagen
image_path = '/home/adrian/Escritorio/Robotica/3º/3_Curso_1_Cuatri/Vision_Artificial/Practicas/P2/Exercicios_clase/flowers.png'
image_color = cv2.imread(image_path, 1)

# Por si no se encuentra la imagen
if image_color is None:
    print('No se encuentra la imagen')
    exit(1)

# Visualizamos la imagen con OpenCV
cv2.namedWindow('Flores a color',cv2.WINDOW_NORMAL)
cv2.imshow('Flores a color',image_color)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Visualizamos la imagen con matplotlib
#plt.imshow(image_color)
#plt.show()

# Cargamos la imagen en tonos de gris y calculamos los datos pedidos
image_gray = cv2.imread(image_path, 0)
media = np.mean(image_gray) # Media
maximo = np.max(image_gray) # Maximo
minimo = np.min(image_gray) # Minimo
desviacion = np.std(image_gray) # Desviacion tipica
print('Media: {}\nMaximo: {}\nMinimo: {}\nDesviacion tipica: {}'.format(media, maximo, minimo, desviacion))

# Añadimos el rectangulo a la imagen y la visualizamos
image_gray[30:55,0:25] = 255
cv2.namedWindow('Flores gris con rectangulo',cv2.WINDOW_NORMAL)
cv2.imshow('Flores gris con rectangulo',image_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Redimensionamos la imagen a 255x255 px y la visualizamos
image_resize = cv2.resize(image_gray, (255, 255))
cv2.namedWindow('Flores gris con rectangulo redimensionada',cv2.WINDOW_NORMAL)
cv2.imshow('Flores gris con rectangulo redimensionada',image_resize)
cv2.waitKey(0)
cv2.destroyAllWindows()
