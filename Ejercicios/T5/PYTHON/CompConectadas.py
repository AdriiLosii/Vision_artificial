import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)
matplotlib.rcParams['image.cmap'] = 'gray'

# Lemos a imaxe como escala de grises
im = cv2.imread("../data/truth.png", cv2.IMREAD_GRAYSCALE)
cv2.imshow("imaxe",im)
cv2.waitKey(0)

# Limiar a imaxe
th, imThresh = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY)

# Atopamos as compoñentes conectadas
_, imLabels = cv2.connectedComponents(imThresh)
plt.imshow(imLabels)
plt.show()

# Visualizamos 
nComponents = imLabels.max()

displayRows = int(np.ceil(nComponents/3.0))
plt.figure(figsize=[20,12])
for i in range(nComponents+1):
    plt.subplot(displayRows,3,i+1)
    plt.imshow(imLabels==i)
    if i == 0:
        plt.title("Fondo, Componhente ID : {}".format(i))
    else:
        plt.title("Componhente ID : {}".format(i))
plt.show()



#A seguinte liña atopa os valores mínimos e máximos de píxeles
# e as súas localizacións na imaxe.
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(imLabels)

# Normaliza a imaxe entre 0 e 255
imLabels = 255 * (imLabels - minVal)/(maxVal-minVal)

# convertemos a imaxe a unsigned  8-bits
imLabels = np.uint8(imLabels)

# Aplicamos un mapa de cor
imColorMap = cv2.applyColorMap(imLabels, cv2.COLORMAP_JET)
cv2.imshow("imaxe",imColorMap)
cv2.waitKey(0)
