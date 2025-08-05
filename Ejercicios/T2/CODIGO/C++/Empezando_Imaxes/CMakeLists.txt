cmake_minimum_required(VERSION 2.8.12)
PROJECT(Empezando_imaxes)
SET(CMAKE_CXX_STANDARD 11)

##########################################################################
#############  Establecemos as regras de compilado e linakdo  ############
##########################################################################

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS})

MACRO(add_ficheiro nome)
  ADD_EXECUTABLE(${nome}.out ${nome}.cpp)
  TARGET_LINK_LIBRARIES(${nome}.out ${OpenCV_LIBS} )
ENDMACRO()


add_ficheiro(Copiar_rexion_a_outra)
add_ficheiro(Crea_unha_Imaxemascara)
add_ficheiro(Corta_unha_seccion_imaxe)
add_ficheiro(redimensionar_imaxe)
add_ficheiro(Crea_novas_imaxes)
add_ficheiro(Separando_e_unindo_canles)
