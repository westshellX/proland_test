SET(HEADER_PATH ${proLandOceanTry_SOURCE_DIR}/OceanLightingFFT )
SET(PROJECTNAME OceanLightingFFT)

INCLUDE_DIRECTORIES(
   ${HEADER_PATH}
   ${anttweakbar_INCLUDE_DIR}
)

# Adds headers to the "Header Files" group in visual studio
SOURCE_GROUP( "Header Files" ${HEADER_PATH}/*.h )

ADD_EXECUTABLE(
  ${PROJECTNAME}
  Program.cpp
  Main.cpp
)
target_link_libraries(
    ${PROJECTNAME} 
	
	${anttweakbar_LIBRARY}
	
	${glew_LIBRARY}
	${GLUT_LIBRARY}
)








