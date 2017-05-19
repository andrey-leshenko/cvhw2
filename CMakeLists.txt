cmake_minimum_required(VERSION 3.2)

project(cvhw2)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include(./dlib/dlib/cmake)

add_executable(cvhw2 cvhw2.cpp list_directory.cpp)
target_link_libraries(cvhw2 ${OpenCV_LIBS} dlib)

if (CMAKE_COMPILER_IS_GNUCC)
	set_property( TARGET cvhw2 APPEND_STRING PROPERTY COMPILE_FLAGS -Wall )
endif (CMAKE_COMPILER_IS_GNUCC)
