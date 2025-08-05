#!/bin/bash

# |             Este script funciona correctamente sobre:      |
# |------------------------------------------------------------|
# | OS               | OpenCV       | Test |   ultimo test     |
# |------------------|--------------|------|-------------------|
# |Ubuntu 22.04  LTS | OpenCV 4.6.0 | OK   |   19 Spt 2022     |
# |----------------------------------------------------------- |


# 1. ACTUALIZAMOS UBUNTU

sudo apt-get -y update
sudo apt-get -y upgrade         # descomenta para actualizar os paquetes instalados
# sudo apt-get -y dist-upgrade  # descomenta para soportar cambios nas versions dos paquetes
sudo apt-get -y autoremove    # descomenta para eliminar os paquetes que xa nos son precisos


# 2. INSTALAMOS AS DEPENDENCIAS PARA OPENCV

# Ferramentas xenericas: compilador, construcion, manexo de repositorios, ...:
sudo apt-get install -y build-essential cmake pkg-config unzip git checkinstall wget


# Para poder usar as caracteristicas de OpenCV GUI: qt5-default
sudo apt-get install -y  libvtk7-dev libgtk2.0-dev libgtk-3-dev libx11-dev mesa-utils freeglut3 freeglut3-dev

# Media I/O:
sudo apt-get install -y zlib1g-dev libjpeg-dev libwebp-dev libpng-dev libtiff5-dev libopenexr-dev libgdal-dev
#libreria libjasper-dev abandonada.


# Video I/O:libavresample-dev
sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libtheora-dev libvorbis-dev libxvidcore-dev libx264-dev yasm libopencore-amrnb-dev libopencore-amrwb-dev libv4l-dev libxine2-dev ibgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libfaac-dev libmp3lame-dev x264 libmp3lame-dev 

#librerias de programacion de interfaces de camaras
sudo apt-get install libdc1394 libdc1394-dev libxine2-dev libv4l-dev v4l-utils
cd /usr/include/linux
sudo ln -s -f ../libv4l1-videodev.h videodev.h
cd ~

# Librerias de paraleliacion e alxebra:
sudo apt-get install -y libtbb-dev libeigen3-dev

#librerias de optimiazacion para opencv
sudo apt-get install -y libatlas-base-dev libblas-dev liblapack-dev gfortran

# Xestion da documentacion:
sudo apt-get install -y doxygen


#librerias opcionais
sudo apt-get install -y libprotobuf-dev protobuf-compiler
sudo apt-get install -y libgoogle-glog-dev libgflags-dev
sudo apt-get install -y libgphoto2-dev libeigen3-dev libhdf5-dev

# Java:
sudo apt-get install -y ant default-jdk

