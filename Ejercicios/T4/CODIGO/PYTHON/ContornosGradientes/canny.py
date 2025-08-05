import cv2
import numpy as np

lowThreshold = 50
highThreshold = 100

maxThreshold = 1000

apertureSizes = [3, 5, 7]
maxapertureIndex = 2
apertureIndex = 0

blurAmount = 0
maxBlurAmount = 20

# Funcion para a chamada do trackbar
def applyCanny():
    # Suavizamos a imaxe antes de detectar os contornos
    if(blurAmount > 0):
        blurredSrc = cv2.GaussianBlur(src,
                        (2 * blurAmount + 1, 2 * blurAmount + 1), 0)
    else:
        blurredSrc = src.copy()

    # Canny require un tamanho de aperture impar!
    apertureSize = apertureSizes[apertureIndex]

    # Aplicamos canny para obter os bordes
    edges = cv2.Canny( blurredSrc,
                        lowThreshold,
                        highThreshold,
                        apertureSize = apertureSize )
    cv2.imshow("Bordes",edges)
    #cv2.waitKey(0)

# Funcion para actualizar o limiar inferior
def updateLowThreshold( *args ):
    global lowThreshold
    lowThreshold = args[0]
    applyCanny()
    pass

# Funcion para actualizar o limiar superior
def updateHighThreshold( *args ):
    global highThreshold
    highThreshold = args[0]
    applyCanny()
    pass

# Funcion para actualizar o suavizado
def updateBlurAmount( *args ):
    global blurAmount
    blurAmount = args[0]
    applyCanny()
    pass

# actualizacion do tamanho de apertura
def updateApertureIndex( *args ):
    global apertureIndex
    apertureIndex = args[0]
    applyCanny()
    pass

# Read sample image
src = cv2.imread('../../data/Contour.png', cv2.IMREAD_GRAYSCALE)
if src is None:
    print("Non poiden ler a imaxe")

edges = src.copy()
# Visualizamos
cv2.namedWindow("Bordes", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Bordes", src)
# Trackbar para controlar o limiar inferior
cv2.createTrackbar( "Limiar inferior", "Bordes", lowThreshold,
            maxThreshold, updateLowThreshold)

# Trackbar para controlar o limiar superior
cv2.createTrackbar( "Limiar superior", "Bordes", highThreshold,
            maxThreshold, updateHighThreshold)

# Trackbar para controlar o tamanho da apertra
cv2.createTrackbar( "Tamanho apertura", "Bordes", apertureIndex,
            maxapertureIndex, updateApertureIndex)

# Trackbar para controlas o suavizado
cv2.createTrackbar( "Suavizado", "Bordes", blurAmount, maxBlurAmount,
            updateBlurAmount)
k = cv2.waitKey(0)

if k == 27:
    cv2.destroyAllWindows()
