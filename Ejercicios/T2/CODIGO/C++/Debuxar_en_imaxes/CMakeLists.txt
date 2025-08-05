cmake_minimum_required(VERSION 2.8.12)
PROJECT(Debuxar_imaxes)
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


add_ficheiro(Debuxamos_texto)
add_ficheiro(D_Circles)
add_ficheiro(D_Elipse)
add_ficheiro(D_Linhas)
add_ficheiro(D_Rectan)

