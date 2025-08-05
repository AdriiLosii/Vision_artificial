#ESTES PASOS DEBENSE INSTALAR MANUALMENTE SE QUERES QUE A TÚA TARXETA NVIDA SEXA EMPREGADA
#COMO UN DISPOSITIVO DE CALCULO. O FORMATO DE SCRIPT DE SHELL SO É PARA SECUENCIAR O ORDE DOS PASOS
# POR EXEMPLO, AS LIBCUDA DEBES IDENTIFICARTE NA WEB DE NVIDIA PARA OBTER AS TÚAS LIGAZÓNS DE DESCARGA.

# ACTUALIZAMOS UBUNTU

sudo apt-get -y update
sudo apt-get -y upgrade         # descomenta para actualizar os paquetes instalados
sudo apt-get -y dist-upgrade  # descomenta para soportar cambios nas versions dos paquetes
sudo apt-get -y autoremove    # descomenta para eliminar os paquetes que xa nos son precisos

#POR SI LAS MOSCAS: https://linuxconfig.org/how-to-install-the-nvidia-drivers-on-ubuntu-22-04

#####################################################################################
#PASO 1: INSTALAMOS O DRIVER DE NVIDIA

# 1.-CAL É A TARXETA DE NVIDA QUE TEN INSTALADA O TEU EQUIPO?, 

# OPCIÓN 1.-INSTALA O DRIVER DE NVIDIA GRAFICAMENTE (METODO GUI).
1.- Abre a aplicación: Actualizacion de software dende o menú das apps
2.- Selecciona a aleta de Mais Controladores
3.- Selecciona o driver de Nvidia desexado
4.- Aplica cambios
5.- Reinicia o ordenador
nvidia-smi #CON ESTE COMANDO VISIALIZAS O ESTADO DA TUA TARXETA


#####################################################################################
#OPCIÓNS 2 E 3  VIA COMANDOS NA CONSOLA

#OPCIÓN 2 ###########################################################################
ubuntu-drivers devices #Executa o seguinte comando na túa consola e dirache que hardware tes e cal e o driver recomendado
sudo ubuntu-drivers autoinstall # INSTALA O DRIVER DE NVIDIA PARA A TÚA TARXETA RECOMENDADO POLO COMANDO ANTERIOR.
sudo reboot #unha vez instalado hai que reiniciar o ordenador!!
nvidia-smi #CON ESTE COMANDO VISIALIZAS O ESTADO DA TUA TARXETA
#####################################################################################

#OPCIÓN 3 ################################################################################
#Alternativamente podes empregar apt-get para instalar a versión doo driver que ti queiras. 
apt search nvidia-driver   #lista os driver dispoñibles para a GPU
# Por exemplo, se quero a version 470 executaría: 
sudo apt install nvidia-driver-470
sudo reboot
nvidia-smi #CON ESTE COMANDO VISIALIZAS O ESTADO DA TUA TARXETA
#####################################################################################


#####################################################################################
#(OLLO: SO SE O PRECISAS) PARA DESINSTALAR TODOS OS PAQUETES DE NVIDIA DO SISTEMA ###
dpkg -l | grep -i nvidia   #COMPROBAMOS QUE PAQUETES DE NVIDIA ESTAN INSTALADOS NO SISTEMA
sudo apt-get remove --purge '^nvidia-.*' #Este comando purga todos os paquetes de Nvidia

#Se o paquete ubuntu-desktop é elimiando, reinstalao co comando
sudo apt-get install ubuntu-desktop
sudo reboot

# OLLO: Se Precisas desinstalar o driver antiguo, executa: sudo apt purge nvidia-driver-XXX
#####################################################################################





#####################################################################################
#PASO 2: INSTALAMOS CUDA TOOLKIT
# Vai a esta dirección e selecciona as opcións en función da túa maquina, tarxeta e SO.
# CUDA: https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu
#Unha vez rematada a elección, darache unha serie de comandoS (como os que están abaixo) que debes executar dende a túa consola.

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.7.1/local_installers/cuda-repo-ubuntu2204-11-7-local_11.7.1-515.65.01-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-11-7-local_11.7.1-515.65.01-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-11-7-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda

#####################################################################################

#####################################################################################
#PASO 3: INSTALAMOS cuDNN
# Vai a dirección de NVIDIA: https://developer.nvidia.com/Cudnn
# para identificarte na web de NVIDIA e selccionar o paquete de cuDNN dexedado.
# compatible coa versión de CUDA que instalaches no paso previo (neste caso CUDA 11.7).
# A continuación, selecciona o instalador local para a túa version de Ubuntu e arquitectura da túa maquina:

wget https://developer.nvidia.com/compute/cudnn/secure/8.5.0/local_installers/11.7/cudnn-local-repo-ubuntu2204-8.5.0.96_1.0-1_amd64.deb

# Unha vez os decarges, executa as ordes de 
sudo apt-get -y install cudnn-local-repo-ubuntu2204-8.5.0.96_1.0-1_amd64.deb
#####################################################################################

#OLLO: ASEGURATE QUE OS PATHS SON OS CORRECTOS PARA A TUA MAQUINA. MODIFICAOS SEGUNDO PROCEDA!!!

#SE TODO FOI BEN, SO NOS QUEDA MODIFICAR O NOSO FICHEIRO DE CONFIGURACIÓN DA BASH PARA QUE
# OS PATH DO NOSO USUARIO ATOPEN ONDE ESTÁ INSTALADO CUDA. )
# OLLO: DEBES ADAPTAR OS PATH A VERSION DE CUDA QUE INSTALES!! NESTE CASO E A CUDA11.1 (COMPROBA QUE EXISTES OS PATHS QUE ESCRIBIREMOS NO .BASHRC NO TEU ORDENADOR
#Modificamos o ficheiro ~/.bashrc
echo "#Cambiamos os pahts pola instalacion de cuda" >> ~/.bashrc
echo "export PATH=/usr/local/cuda-12.3/bin:$PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64:$LD_LIBRARY_PATH" >> ~/.bashrc
echo "export CUDA_HOME=/usr/local/cuda-12.3" >> ~/.bashrc

