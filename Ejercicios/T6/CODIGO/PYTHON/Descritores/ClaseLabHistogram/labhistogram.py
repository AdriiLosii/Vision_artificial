# importamos paquetes
import cv2

class LabHistogram:
	def __init__(self, bins):
		# Almacenamos os bins que queremos para o histograma
		self.bins = bins

	def describe(self, image, mask=None):
		# Pasamos ao espazo de cor L*a*b*, achamos o histograma e normalizamolo
		lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
		hist = cv2.calcHist([lab], [0, 1, 2], mask, self.bins,
			[0, 256, 0, 256, 0, 256])

		# Histograma para versión de OpenCV 3+
		hist = cv2.normalize(hist,hist).flatten()

		# devolvemos o histograma
		return hist
