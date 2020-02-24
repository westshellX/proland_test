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
	
	debug ${anttweakbar_LIBRARY}
	optimized ${anttweakbar_LIBRARY}
	
	${glew_LIBRARY}
	${glut_LIBRARY}
)
install(
    TARGETS ${PROJECTNAME}
	RUNTIME DESTINATION ${EXECUTABLE_OUTPUT_PATH}
)







