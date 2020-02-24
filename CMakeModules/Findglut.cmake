# - Try to find freeglut
#
#  freeglut_FOUND - system has freeglut
#  glut_INCLUDE_DIR - the freeglut include directories
#  glut_LIBRARY - link these to use freeglut

FIND_PATH(
  glut_INCLUDE_DIR
  NAMES GL/freeglut.h
  PATHS include
)

FIND_LIBRARY(
  glut_LIBRARY
  NAMES freeglut32 freeglut
  PATHS lib
)

IF(glut_INCLUDE_DIR AND glut_LIBRARY)
  SET(freeglut_FOUND TRUE)
ENDIF(glut_INCLUDE_DIR AND glut_LIBRARY)

IF(freeglut_FOUND)
   IF(NOT freeglut_FIND_QUIETLY)
      MESSAGE(STATUS "Found freeglut: ${glut_LIBRARY}")
   ENDIF(NOT freeglut_FIND_QUIETLY)
ELSE(freeglut_FOUND)
message(STATUS "freeglut NOT found")
   IF(freeglut_FIND_REQUIRED)
      MESSAGE(FATAL_ERROR "Could not find freeglut")
   ENDIF(freeglut_FIND_REQUIRED)
ENDIF(freeglut_FOUND)
