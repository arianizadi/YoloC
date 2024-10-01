cmake_minimum_required(VERSION 3.10)

project(YoloC)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Add your executables
add_executable(video video.cpp)
add_executable(live live.cpp)

# Link OpenCV libraries
target_link_libraries(video ${OpenCV_LIBS})
target_link_libraries(live ${OpenCV_LIBS})
