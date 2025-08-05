cmake_minimum_required(VERSION 2.8.12)
PROJECT(Clasificacion)
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


add_ficheiro(Adestra_detector_peons)
add_ficheiro(Clsificador_gafas)
add_ficheiro(Demo_detect_caras)
add_ficheiro(detector_peons)
add_ficheiro(detector_sorrisos)



