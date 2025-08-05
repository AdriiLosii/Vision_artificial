cmake_minimum_required(VERSION 2.8.12)
PROJECT(seguimento_video)
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


add_ficheiro(camShift_tracker)
add_ficheiro(kalmanTracker)
add_ficheiro(LK_tracker)
add_ficheiro(meanShift_tracker)
add_ficheiro(multitracker_opcv)
add_ficheiro(simple_tracker_opcv)



