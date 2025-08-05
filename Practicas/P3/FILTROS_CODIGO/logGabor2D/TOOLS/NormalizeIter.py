# Xose R. Fdez-Vidal (e-correo: xose.vidal@usc.es)
# Codigo docente para o grao de Robótica da EPSE de Lugo.
# Dept. Física Aplicada, Universidade de Santiago de Compostela, 
# GALIZA,  2022 

# Implementamos o algoritmo de normalizacion iterativa
# descrito en:
# L. Itti, C. Koch, A saliency-based search mechanism for overt 
# and covert shifts of visual attention, Vision Research, 
# Vol. 40, No. 10-12, pp. 1489-1506, May 2000.
#Esta normalización potencia os maximos mais potentes e vai
#eliminando os mais debiles. 
# OLLO: considera que os máximos a buscar son de luminancia (brancos)
#Se desexaramos achar os mínimos tamén nos serviría invertindo a imaxe de entrada

#Importamos as librerias
import argparse
import cv2
import numpy as np


def normalizeIterGauss(data, niter =2, minmax = (0,10)):
    #Parametros para a construccion dos filtros
    iterInhi = 2.0
    iterCoEx = 0.5
    iterCoIn = 1.5
    iterExSig = 2
    iterInSig = 25

    #Normalizamos a imaxe ao rango indicado
    result = np.float32(cv2.normalize(np.float32(data).clip(0),
                        None, minmax[0], minmax[1], cv2.NORM_MINMAX))

    #Construimos os kernels gaussianos 1D para excitacion e inhibicion
    sz = np.max(result.shape)
    maxhw = int(np.maximum(0, np.floor(np.min(result.shape)/2) - 1))
    esig = sz * iterExSig * 0.01
    isig = sz * iterInSig * 0.01

    #Construimos os kernels gaussianos 1D: para imaxes con blobs blancos sobre fondo escuro mellor non normalizar
    gExc = (iterCoEx)*np.float32(cv2.getGaussianKernel(maxhw,esig))
    gInh = (iterCoIn)*np.float32(cv2.getGaussianKernel(maxhw,isig))

    #iteramos ...
    for i_iter in range(niter):
    
        # Aplicamos os kernels gaussianos 1D a X e Y.
        # O filtro gaussiano e separable e pasamos de unha
        # complexidade O(n^2) en 2D a O(n) co filtro separable
        excit = cv2.sepFilter2D(result,-1,gExc,gExc)
        inhib = cv2.sepFilter2D(result,-1,gInh,gInh)

        # inhibicion global para previr a explosion
        # do mapa de actividade
        globinhi = 0.01 * iterInhi * np.max(result)

        # combinamos todo e recortamos valores inferiores a cero
        result = (result + excit - inhib - globinhi).clip(0)

    #Normalizamos o resultado entre [0,1]
    result = np.float32(cv2.normalize(np.float32(result), None, 0, 1, cv2.NORM_MINMAX))
    return result


def main(args):
    #Lemos a imaxe en formato gris e visualizamos
    img = cv2.imread(args['image'], 0)
    if img is None:
        raise Exception("Non atopo a imaxe no sitio indicado")
    cv2.imshow("Orixinal",img)
    cv2.waitKey(0)

    result = normalizeIterGauss(img,niter=3,minmax=(0,10))
    cv2.imshow("Potenciacion Maximos",result)
    cv2.waitKey(0)

    cv2.destroyAllWindows()



if __name__ == '__main__':
    # analizamos os argumentos de entrada
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Imaxe de entrada")
    args = vars(ap.parse_args())
    main(args)