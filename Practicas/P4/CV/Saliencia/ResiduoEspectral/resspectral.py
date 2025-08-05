import numpy as np
import cv2
from superpos_mapa import superpos_colormap


WIDTH = 64   # has a great influence on the result

if __name__ == '__main__':
	img = cv2.imread('test1.png', 0)
	rows, cols = img.shape
	img = cv2.resize(img, (WIDTH,int(WIDTH*img.shape[0]/img.shape[1])))

	c = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)	# Discrete Fourier Transform: pasar de dominio del espacio al de frecuencia
	mag = np.sqrt(c[:,:,0]**2 + c[:,:,1]**2)	# Modulo (magnitud)
	spectralResidual = np.exp(np.log(mag) - cv2.boxFilter(np.log(mag), -1, (3,3)))

	c[:,:,0] = c[:,:,0] * spectralResidual / mag
	c[:,:,1] = c[:,:,1] * spectralResidual / mag
	c = cv2.dft(c, flags = (cv2.DFT_INVERSE | cv2.DFT_SCALE))
	mag = c[:,:,0]**2 + c[:,:,1]**2
	#cv2.normalize(cv2.GaussianBlur(mag,(9,9),3,3), mag, 0., 1., cv2.NORM_MINMAX)
	# sm= cv2.normalize(cv2.GaussianBlur(mag,(9,9),3,3), mag, 0, 255, cv2.NORM_MINMAX)
	# cv2.imshow('Saliency Map', mag)
	sm = cv2.resize(cv2.GaussianBlur(mag,(9,9),3,3), (cols, rows), cv2.INTER_LANCZOS4)
	sm = (cv2.normalize(sm, mag, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8)

	
	###################################################
    # Visualización del mapa de saliencia final
    ###################################################
	im_or_gray= cv2.imread('test1.png',0)
	im_or = cv2.imread('test1.png',1)
	sm_cor = np.zeros(im_or.shape).astype(np.uint8)
	for i in range(3):
		sm_cor[:,:,i]=sm
	superpos_colormap(im_or, sm)
	superpos_colormap(im_or_gray, sm)
	c = cv2.waitKey(0) & 0xFF
	if(c==27 or c==ord('q')):
		cv2.destroyAllWindows()
