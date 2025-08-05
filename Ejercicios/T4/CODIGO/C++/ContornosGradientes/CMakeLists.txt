cmake_minimum_required(VERSION 2.8.12)
PROJECT(Contornos_basados_gradientes)
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


add_ficheiro(canny)
add_ficheiro(contornos)
add_ficheiro(enfoque)
add_ficheiro(laplaciana)
add_ficheiro(sobel)


