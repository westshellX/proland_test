# - Try to find glew
#
#  glew_FOUND - system has glew
#  glew_INCLUDE_DIR - the glew include directories
#  glew_LIBRARY - link these to use glew

FIND_PATH(
  glew_INCLUDE_DIR
  NAMES GL/glew.h
  PATHS include
)

FIND_LIBRARY(
  glew_LIBRARY
  NAMES glew32 glew
  PATHS lib
)

IF(glew_INCLUDE_DIR AND glew_LIBRARY)
  SET(glew_FOUND TRUE)
ENDIF(glew_INCLUDE_DIR AND glew_LIBRARY)

IF(glew_FOUND)
   IF(NOT glew_FIND_QUIETLY)
      MESSAGE(STATUS "Found glew: ${glew_LIBRARY}")
   ENDIF(NOT glew_FIND_QUIETLY)
ELSE(glew_FOUND)
message(STATUS "glew NOT found")
   IF(glew_FIND_REQUIRED)
      MESSAGE(FATAL_ERROR "Could not find glew")
   ENDIF(glew_FIND_REQUIRED)
ENDIF(glew_FOUND)
