cmake_minimum_required(VERSION 3.10)

project(YoloC)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Add your executable
add_executable(yoloc main.cpp)

# Link OpenCV libraries
target_link_libraries(yoloc ${OpenCV_LIBS})

