# Xose R. Fdez-Vidal (e-correo: xose.vidal@usc.es)
# Codigo docente para o grao de Robótica da EPSE de Lugo.
# Dept. Física Aplicada, University of Santiago, 
# GALIZA,  2022 

"""
Este programa esta feito para ilustrar como facer un suavizado 3D
construindo unha gaussiana directamente no dominio de Fourier.
A trasnformada de Fourier dunha gaussiana é outra gaussiana coa
seguinte función:
g = np.exp(-2.0*np.pi**2*((u**2)*sigma[0] + (v**2)*sigma[1]+ + (w**2)*sigma[3]))

"""

import cv2
import argparse
import scipy.io as sio
import numpy as np
from mayavi.mlab import *
from pyfftw.interfaces.scipy_fftpack import fftn, ifftn, fftshift



def f_gauss3D(size, sigma):

    """Funcion para construir o kernel da TF dunha gaussiana 3D 
        directamente no dominio da frecuencia
    """
    rows, cols, slices = size
    u, v, w= np.meshgrid(np.linspace(-0.5, 0.5, cols, endpoint=(cols % 2)),
                         np.linspace(-0.5, 0.5, rows, endpoint=(rows % 2)),
                         np.linspace(-0.5, 0.5, slices, endpoint=(slices % 2)),
                         sparse = True)
    g = np.exp(-2.0*np.pi**2*((u**2)*sigma[0] + 
                              (v**2)*sigma[1] + 
                              (w**2)*sigma[2]))
    #Para visualizar executa: ipython --gui=qt e logo %run G3D_f.py --vid nome video
    #Ollo: as tres dimensions deben ser iguais senon contour3d falla
    #obj = contour3d(u, v, w, g, contours=6, transparent=True)
    return g

def suaviza_G3D(par_vid,sigmas,video_npy):
    #Creamos un kernel Gaussiano do mesmo tamanho que o volume de frames
    kernel3D = f_gauss3D((par_vid[1],par_vid[0],par_vid[2]), (sigmas[0],sigmas[1],sigmas[2]))

    #Convolucionamos
    result = np.real((ifftn(fftshift(kernel3D)*fftn(video_npy))))
    return result

def ler_frames_video(nome_vid):

    #Lemos o video
    video_cap = cv2.VideoCapture(nome_vid)
    if (video_cap.isOpened() == False):
        print("Error na apertura do ficheiro {}".format(nome_vid))
    
    # Recuperamos as propieades do video.
    frame_w   = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h   = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    number_of_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    par_vid = (frame_w, frame_h, number_of_frames, frame_fps)
    print('frame_w = {}, frame_h={} numero frames {} fps {}'.format(frame_w, frame_h, number_of_frames, frame_fps))

    vid = np.zeros((frame_h,frame_w,number_of_frames))
    nf = 0
    while True:
        # Lemos todos os frames.
        ok, frame = video_cap.read()
        if not ok:
            break
        vid[:,:,nf]=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        nf +=1
    return vid, par_vid

def main(args):

    print(args.vid)
    #Lemos o vido
    vid, par_vid = ler_frames_video(args.vid)

    #Configuramos a escritura do video
    # Especificamos os valores para  fourcc.
    #fourcc_avi = cv2.VideoWriter_fourcc('M','J','P','G')
    fourcc_mp4 = cv2.VideoWriter_fourcc(*'mp4v')
    file_out = 'vid_smooth.mp4'
    out = cv2.VideoWriter(file_out, fourcc_mp4, par_vid[3], (par_vid[0],par_vid[1]))

    vid_out = suaviza_G3D(par_vid,(50,50,50),vid)

    #Salvamos o video suavizado a disco
    for idxframe in range(par_vid[2]):
        tmp = (np.floor(cv2.normalize(vid_out[:,:,idxframe], None, 0, 255, cv2.NORM_MINMAX))).astype(np.uint8)
        tmp = cv2.merge([tmp, tmp, tmp])
        out.write(tmp)
    out.release()


if __name__ == '__main__':
    # argparse
    parser = argparse.ArgumentParser(description="Suavizado Gaussiano 3D.",
                                     usage="G3D_F.py -i <input_path>")
    parser.add_argument("-v", "--vid", type = str, help = "nome do video")
    #parser.print_help()
    args = parser.parse_args()
    main(args)