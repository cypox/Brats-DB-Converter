cmake_minimum_required(VERSION 2.8)

# ---[ project
project(DBGenerator)
add_definitions(-DCPU_ONLY)
#set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -std=c++11)


# ---[ ITK
set(ITK_DIR "/opt/ITK/Bin")
find_package(ITK REQUIRED)
if (ITK_FOUND)
    include(${ITK_USE_FILE})
endif(ITK_FOUND)


# --[ Caffe
find_package(Caffe)
include_directories(${Caffe_INCLUDE_DIRS})


# ---[ compile
add_executable(main main.cpp)
add_executable(generate generate_raw.cpp)


# ---[ link
target_link_libraries(main ${ITK_LIBRARIES} ${Caffe_LIBRARIES})
target_link_libraries(generate ${ITK_LIBRARIES})
