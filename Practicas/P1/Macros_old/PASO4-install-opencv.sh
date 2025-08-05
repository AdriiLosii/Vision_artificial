#!/bin/bash

# |             Este script funciona correctamente sobre:      |
# |------------------------------------------------------------|
# | OS               | OpenCV       | Test |   ultimo test     |
# |------------------|--------------|------|-------------------|
# |Ubuntu 20.04  LTS | OpenCV 4.5.0 | OK   |   11 Nov 2020     |
# |----------------------------------------------------------- |



#--------------------------------------------------------------------------------------------------------------------------------
#PARA INSTALAR OPNCV SOBRE UBUNTU HAI DÚAS OPCIÓNS:
#Opcion 1. Instalar OpenCV dende o repositorio de Ubuntu executando os comandos:
# sudo apt-get install libopencv-dev  #para instalar as librarias de C++
# sudo apt-get install python3-opencv # para instalar OpCV para python no raiz do sistema
# No caso de que quiras instalar openCv para Python no teu entorno virutal de Python soamente podes empregar pip install opencv-contrib-python
# Sen embargo, esta opción ten desvantaxes:
#	-Non terás a última versión de OpenCV instalada
#	-Non teras todas as opción que desexes en opencv: algoritmos privados e, sobre todo,
#	 sen soporte para CUDA (emprego da tarxeta grafica NVIDA para acelerar os cálculos)
#
#
#Opción 2. Instalar OpenCV a partir do repositorio oficial de OpenCV e compilar o código fonte coas opcións que desexes. Para iso,
#Paso 1.- asegurate de desintalar calquera version anterior de OpenCV instalada con anterioridade no teu sistema dende os repositorios (sudo apt-get install libopencv-dev python3-opencv) e se foi compilada, enton vai a onde tes o código fonte, ao directorio build e, dende aí, executa sudo make uninstall
#Paso 2.- Executa este script na consola: bash install-opencv.sh
#Isto executara todos os pasos necesarios por ti e automaticamente.
#
#Introduce a contrasinal de sudo cando cha pida e tómate un café.
# Tardara tempo (moito tempo se a túa máquina e a túa conexión a internet son lentas)
#Referencia:
#https://docs.opencv.org/master/d2/de6/tutorial_py_setup_in_ubuntu.html

# ----------------------------------------------------------------------------------------------------------------------------- |
#                                              OPCIÓNS DO SCRIPT OPENCV (C++/PYTHON) CON CUDA                                   |
# ------------------------------------------------------------------------------------------------------------------------------|
OPENCV_VERSION='4.8.0'       	# version para ser instalada de OpenCV
OPENCV_CONTRIB='SI'         	# Instalamos os módulos extras de Opencv (SI/NON)

PYTHON_VERSION='python3.8'	# Version python instalada no sistema
NOME_ENTORNO_PYTHON='VA'    	# nome do entorno virtual de python onde instalaremos openc

GCC_VERSION='gcc-9'		# versión do compilador c/c++

# ---------------------------------------------------------------------------------------------------------------------------- |


# 1. CONSTRUIMOS O PAQUETE OPENCV E INSTALAMOLO NO ENTORNO VIRTUAL DE PYTHON SELECCIONADO
# ASEGURATE DE QUE NON HAI UN DIRECTORIO NO TEU HOME CO NOME DE tmp!!!!!!

cd ~ && mkdir tmp && cd tmp && mkdir OpenCV  #criamos un directorio ~/tmp para este fin
wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip
unzip ${OPENCV_VERSION}.zip && rm ${OPENCV_VERSION}.zip
mv ./opencv-${OPENCV_VERSION}/* OpenCV
rm -rf opencv-${OPENCV_VERSION}

if [ $OPENCV_CONTRIB = 'SI' ]; then
  wget https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip
  unzip ${OPENCV_VERSION}.zip && rm ${OPENCV_VERSION}.zip
  mv opencv_contrib-${OPENCV_VERSION} opencv_contrib
  mv opencv_contrib OpenCV
fi



cd OpenCV && mkdir build && cd build

if [ $OPENCV_CONTRIB = 'NON' ]; then
cmake	-D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_C_COMPILER=/usr/bin/${GCC_VERSION} -D WITH_QT=ON -D WITH_GTK=ON -D WITH_OPENGL=ON -D FORCE_VTK=ON\
	-D WITH_XINE=ON -D WITH_GDAL=ON -D CMAKE_INSTALL_PREFIX=/usr/local \
	-D INSTALL_PYTHON_EXAMPLES=ON -D INSTALL_C_EXAMPLES=ON -D WITH_TBB=ON -D WITH_CUDA=OFF -D BUILD_opencv_cudacodec=OFF \
	-D ENABLE_FAST_MATH=1 -D CUDA_FAST_MATH=1 -D WITH_CUBLAS=1 -D WITH_V4L=ON  -D OPENCV_GENERATE_PKGCONFIG=ON\
	-D WITH_GSTREAMER=ON -D OPENCV_PC_FILE_NAME=opencv.pc -D OPENCV_ENABLE_NONFREE=ON \
	-D OPENCV_PYTHON3_INSTALL_PATH=~/.virtualenvs/${NOME_ENTORNO_PYTHON}/lib/${PYTHON_VERSION}/site-packages \
	-D PYTHON_EXECUTABLE=~/.virtualenvs/${NOME_ENTORNO_PYTHON}/bin/python -D BUILD_EXAMPLES=ON ..
fi


if [ $OPENCV_CONTRIB = 'SI' ]; then

cmake	-D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_C_COMPILER=/usr/bin/${GCC_VERSION} -D WITH_QT=ON WITH_GTK=ON -D WITH_OPENGL=ON -D FORCE_VTK=ON\
	-D WITH_XINE=ON -D WITH_GDAL=ON -D CMAKE_INSTALL_PREFIX=/usr/local \
	-D INSTALL_PYTHON_EXAMPLES=ON -D INSTALL_C_EXAMPLES=ON -D WITH_TBB=ON -D WITH_CUDA=OFF -D BUILD_opencv_cudacodec=OFF \
	-D ENABLE_FAST_MATH=1 -D CUDA_FAST_MATH=1 -D WITH_CUBLAS=1 -D WITH_V4L=ON  -D OPENCV_GENERATE_PKGCONFIG=ON\
	-D WITH_GSTREAMER=ON -D OPENCV_PC_FILE_NAME=opencv.pc -D OPENCV_ENABLE_NONFREE=ON \
	-D OPENCV_PYTHON3_INSTALL_PATH=~/.virtualenvs/${NOME_ENTORNO_PYTHON}/lib/${PYTHON_VERSION}/site-packages \
	-D PYTHON_EXECUTABLE=~/.virtualenvs/${NOME_ENTORNO_PYTHON}/bin/python -D BUILD_EXAMPLES=ON \
	-D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules ..
fi

#compilamos os fontes, linkamos e instalamos no sistema o paquete OpenCV
#o número 8 ten que ver co número de procesadores que podes adicar a compilación.
#Depende do teu procesador. Para a saber candos podes adicar, executa nproc na túa consola. O número devolto o asignas a j.
make -j16
sudo make install

#Se queremos desintalar Opencv compilado do sistema, imos ao directorio de build e executamos
#sudo make uninstall

#Incluimos as librarias no entorno do teu sistema
sudo /bin/bash -c 'echo "/usr/local/lib" >> /etc/ld.so.conf.d/opencv.conf'
sudo ldconfig

#Se queres ter dispoñibles os enlaces a OpenCV no sistema global deberias copiar o directorio creado durante a instalaciom de OpenCV (
#-DOPENCV_PYTHON3_INSTALL_PATH=~/.virtualenvs/${NOME_ENTORNO_PYTHON}/lib/${PYTHON_VERSION}/site-packages) no cartafol dist-packages do teu interprete de python.

#sudo cp -r ~/.virtualenvs/${NOME_ENTORNO_PYTHON}/lib/${PYTHON_VERSION}/site-packages /usr/local/lib/${PYTHON_VERSION}/dist-packages
#echo "Modificamos o config-3.8.py para apuntar ao cartafol destino"
#sudo nano /usr/local/lib/${PYTHON_VERSION}/dist-packages/cv2/config-3.8.py
#```
#    PYTHON_EXTENSIONS_PATHS = [
#    os.path.join('/usr/local/lib/python3.8/dist-packages/cv2', 'python-3.8')
#    ] + PYTHON_EXTENSIONS_PATHS
#```

# 4. EXECUTAMOS ALGÚN EXEMPLOS DE C++ E PYTHON CON OPENCV2 (abre unha consola e escribe os comandos indicados)

#executamos un programa compilado en C++ durante a compilación de OpenCV
#cd ~/tmp/OpenCV/build/bin
#./example_cpp_edge ../../samples/data/fruits.jpg

#agora repetimos a acción cun programa en Python (asegurate de que tes activado o flag de construir os exemplos en python)
#cd ~/tmp/OpenCV/samples/python
#python3 video.py

#Ou teclea o seguinte na consola:
#workon VAA (OLLO, este é o entorno que crea o script de instalación por defecto. Depende do valor que queiras poñer ti!!)
#python
#import cv2
#print(cv2.__version__)
