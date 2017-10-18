cmake_minimum_required(VERSION 2.8)

project(ViewAndTime)

set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

find_package(OpenCV  REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

INCLUDE_DIRECTORIES(

    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
)


set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library( SLIC SHARED

src/SLIC/slic.cpp
src/SLIC/slic_tool.cpp
)

target_link_libraries(SLIC
    ${OpenCV_LIBS}
)

add_library( ShapeMatch SHARED
    src/ShapeMatch/shapematch.cpp
#    src/ShapeMatch/shape_tool.cpp
)
target_link_libraries(ShapeMatch
    ${OpenCV_LIBS}
)

add_library( AF SHARED

    src/AF/anisometric.cpp
 )
target_link_libraries(AF
    ${OpenCV_LIBS}
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/)


add_executable(run src/main.cpp )
target_link_libraries(run SLIC AF ShapeMatch)


#add_executable(test src/test.cpp)
#target_link_libraries(test ${OpenCV_LIBS})
