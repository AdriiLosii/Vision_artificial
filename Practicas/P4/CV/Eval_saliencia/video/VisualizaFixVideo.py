# Xose R. Fdez-Vidal (e-correo: xose.vidal@usc.es)
# Codigo docente para o grao de Robótica da EPSE de Lugo.
# Dept. Física Aplicada, University of Santiago, 
# GALIZA,  2022 

"""
Este programa esta feito para ilustrar como se poden ler as
fixacions da estrutura de datos FixVideoData.mat de Videos do CiTIUS (somanete para seis videos).
Esta estrutura é un diccinario cunha serie de campos que nos permiten acceder ao video, nome, frame e 
fixacions para cada frame.
"""

import cv2
import numpy as np
#import mat73
import scipy.io as sio


#Modifica isto segundo a tua estrutura de ficheiros
path_to_fixdata_mat = "./fixacions/" #path ata o ficheiro FixVideoData.mat
path_to_video = "./videos/"  #path ao directorio onde estan os videos
path_to_out = "./output/"

#Lemos a estrutra .mat das fixacions. Na variable fixacions poderemos
#acceder asos datos das fixacions.
#fix_dict['EyeTrackVDB'][#video]['Frame'][0]['FixL'][0,#frame][#nunfix,::2 #coordendas x e y de cada fixacion]
#fix_dict = mat73.loadmat(path_to_fixdata_mat + 'FixVideoData.mat')
fix_dict =  sio.loadmat(path_to_fixdata_mat + 'FixVideoData.mat')

#Lazo para todos os videos da base de datos: 
# fix_dict['EyeTrackVDB'].shape[0]
for ivideo in range(fix_dict['EyeTrackVDB'].shape[0]):
    # Lemos o video
    name_video = fix_dict['EyeTrackVDB'][ivideo]['Name'][0][0]
    input_video = path_to_video + name_video 
    video_cap = cv2.VideoCapture(input_video)

    if (video_cap.isOpened() == False):
        print("Error na apertura do ficheiro {}".format(input_video))

    # Recuperamos as propieades do video.
    frame_w   = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h   = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    print('frame_w = {}, frame_h={} fps {}'.format(frame_w,frame_h,frame_fps))

    # Especificamos os valores para  fourcc.
    fourcc_avi = cv2.VideoWriter_fourcc('M','J','P','G')
    fourcc_mp4 = cv2.VideoWriter_fourcc(*'mp4v')
    # enlentecemos a velocidade dos videos grabados 
    # para visualizacion con VLC.
    # Se o consideras oportuno podes deixalo ao seu 
    # framerate orixinal para un resultado normal
    frame_fps = int(frame_fps/3)

    # Especificamos os nomes de saida:
    # file_out_fixCirc --> video coa imaxe orixinal superpora a fixacion cun circulo vermello
    # file_out_fix  --> video que conten as fixacion como puntos na imaxe binaria {0,255}
    # file_out_fixDen --> video que conten o mapa de densidade de fixacions
    
    # file_out_avi = 'video_out.avi'
    file_out_fixCirc = path_to_out + name_video + '_fixCirc.mp4'
    file_out_fix = path_to_out + name_video + '_fix.mp4'
    file_out_fixDen = path_to_out + name_video + '_fixDen.mp4'

    # Creamos os obxectos para escribir o video (avi ou mp4)
    # out_avi = cv2.VideoWriter(file_out_avi, fourcc_avi, frame_fps, (frame_w,frame_h))
    out_fixCirc = cv2.VideoWriter(file_out_fixCirc, fourcc_mp4, frame_fps, (frame_w,frame_h))
    out_fix = cv2.VideoWriter(file_out_fix, fourcc_mp4, frame_fps, (frame_w,frame_h))
    out_fixDen = cv2.VideoWriter(file_out_fixDen, fourcc_mp4, frame_fps, (frame_w,frame_h))

    # Lemos todos os frames do video
    frame_count = 0
    while True:
        # Lemos un frame cada instante.
        ok, frame = video_cap.read()
        if not ok:
            break
        #lemos as fixacion do fotograma e pintamos os circulos
        fix = fix_dict['EyeTrackVDB'][ivideo]['Frame'][0]['FixL'][0,frame_count][:,0:2]
        # Para construir o mapa de densidade de fixacións
        map_fix_frame = np.zeros((frame_h,frame_w))
        for ifix in range(fix.shape[0]):
            #Pode haber fixación fora dos marxes da imaxe!!
            if int(fix[ifix][0]) < frame_h and int(fix[ifix][1]) <frame_w:
                #pintamos un circulo en cada fixacion
                cv2.circle(frame,(int(fix[ifix][0]), int(fix[ifix][1])), 5, (0,0,255))
                #Mapa de fixacions
                map_fix_frame[int(fix[ifix][1]), int(fix[ifix][0])] = 255
        #Convolucionamos cunha gaussiana as fixacion para obter o
        # mapa de densidade de fixacions
        map_den_fix = (np.floor(cv2.normalize(cv2.GaussianBlur(np.float32(map_fix_frame),(85,85),20,20), None, 0, 255, cv2.NORM_MINMAX))).astype(np.uint8)
        
        # Visualizacion para efectos de depuracion
        cv2.imshow('Superpos Fixacions',frame)
        cv2.imshow('Mapa Densidade de Fixacions',map_den_fix)
        cv2.imshow('Mapa de fixacions',map_fix_frame)
        cv2.waitKey(2)
        
        # Incrementamos o contador de frames para a anotacion.
        frame_count += 1

        # Escribimos os resultados en formato video
        # que se levaran todos ao cartafol output.
        # out_avi.write(frame)
        out_fixCirc.write(frame)
        
        #Formamos un frame coas tres bandas iguais (require imaxe cor)
        tmp = np.zeros_like(frame)
        map_fix_frame=map_fix_frame.astype(np.uint8)
        tmp=cv2.merge([map_fix_frame, map_fix_frame, map_fix_frame])
        out_fix.write(tmp)
        #Formamos un frame coas tres bandas iguais (require imaxe cor)
        tmp = np.zeros_like(frame)
        tmp=cv2.merge([map_den_fix, map_den_fix, map_den_fix])
        out_fixDen.write(tmp)

    # Liberamos memoria e pechamos os fluxos de video
    #out_avi.release()
    video_cap.release()
    out_fixCirc.release()
    out_fix.release()
    out_fixDen.release()
