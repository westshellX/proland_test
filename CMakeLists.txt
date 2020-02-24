cmake_minimum_required(VERSION 2.6)

project(proLandOceanTry)

SET(PRO_VERSION_MAJOR 0)
SET(PRO_VERSION_MINOR 0)
SET(PRO_VERSION "${PRO_VERSION_MAJOR}.${PRO_VERSION_MINOR}")

# Avoid source tree pollution
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
	message(FATAL_ERROR "In-source builds are not permitted. Make a separate folder for building:\nmkdir build; cd build; cmake ..\nBefore that, remove the files already created:\nrm -rf CMakeCache.txt CMakeFiles")
endif(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)


set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

SET(CMAKE_USE_RELATIVE_PATHS true)
set(CMAKE_DEBUG_POSTFIX d)

#查找模式
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/CMakeModules")


find_package(glew REQUIRED)
INCLUDE_DIRECTORIES(${glew_INCLUDE_DIR})

find_package(GLUT)
INCLUDE_DIRECTORIES(${GLUT_INCLUDE_DIR})

find_package(OpenGL)
find_package(anttweakbar)
include_directories(${anttweakbar_INCLUDE_DIR})

find_package(TIFF)

# Sub dirs
add_subdirectory(OceanLighting)
add_subdirectory(OceanLightingFFT)
add_subdirectory(PrecomputedAtmosphericScattering)

