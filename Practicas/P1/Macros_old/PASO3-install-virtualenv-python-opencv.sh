#!/bin/bash

# |             Este script funciona correctamente sobre:      |
# |------------------------------------------------------------|
# | OS               | OpenCV       | Test |   ultimo test     |
# |------------------|--------------|------|-------------------|
# |Ubuntu 20.04  LTS | OpenCV 4.5.3 | OK   |   1 Outubro 2021  |
# |----------------------------------------------------------- |

#---------------------------------------------------------------------------------------
NOME_ENTORNO_PYTHON='VAA'    	# nome do entorno virtual de python onde instalaremos opencv
				#Podes poñer o nome que desexes
#------------------------------------------------------------------------------------------

# 1. ACTUALIZAMOS UBUNTU

sudo apt-get -y update
sudo apt-get -y upgrade         # descomenta para actualizar os paquetes instalados
sudo apt-get -y dist-upgrade  # descomenta para soportar cambios nas versions dos paquetes
sudo apt-get -y autoremove    # descomenta para eliminar os paquetes que xa nos son precisos



#2. Librerias para Python e entornos virtuais:
sudo apt-get install -y python3-dev python3-pip
sudo -H pip3 install -U pip numpy
sudo apt install -y python3-testresources

#creamos instalamos o necesario para crear entornos virtuais de python3
sudo pip install virtualenv virtualenvwrapper
sudo rm -rf ~/.cache/pip
echo "Editamos ~/.bashrc"
echo -e "\n# virtualenv e virtualenvwrapper" >> ~/.bashrc
echo "export WORKON_HOME=$HOME/.virtualenvs" >> ~/.bashrc
echo "export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3" >> ~/.bashrc
echo "source /usr/local/bin/virtualenvwrapper.sh" >> ~/.bashrc


#Para ques este comando funcione o comando source ~/.bashrc en modo non-interactivo (como é este o caso), debes
# comomentar no teu .bashrc as seguintes instruccions
# If not running interactively, don't do anything
#case $- in
#    *i*) ;;
#      *) return;;
#esac
#ou alternativamente invocar o modo interactica cando chamas o script
# bash -i PASO3-install-virtualenv-python-opencv.sh
source ~/.bashrc

#Creamos o entorno virtual chamado segundo o especificado na variable $NOME_ENTORNO_PYTHON
mkvirtualenv ${NOME_ENTORNO_PYTHON} -p python3
#activamos este entorno para instalar opencv nel. O comando deactivate desactiva o entorno activo
workon ${NOME_ENTORNO_PYTHON} 
#instalamos paquetes necesarios
pip install wheel numpy scipy matplotlib scikit-image scikit-learn ipython dlib imutils
pip install pyqtgraph pyqt5 pyopengl
pip install -U Sphinx



