cmake_minimum_required(VERSION 2.8)
project( video_red_channel )
find_package( OpenCV REQUIRED )
add_executable( video_red_channel video_red_channel.cpp )
target_link_libraries( video_red_channel ${OpenCV_LIBS} )
set(OpenCV_DIR /home/josh/Documents/opencv_working/opencv/build/OpenCVConfig.cmake)
