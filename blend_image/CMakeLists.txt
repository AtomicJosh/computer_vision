cmake_minimum_required(VERSION 2.8)
project( blend_image )
find_package( OpenCV REQUIRED )
add_executable( blend_image blend_image.cpp )
target_link_libraries( blend_image ${OpenCV_LIBS} )
set(OpenCV_DIR /home/josh/Documents/opencv_working/opencv/build/OpenCVConfig.cmake)
