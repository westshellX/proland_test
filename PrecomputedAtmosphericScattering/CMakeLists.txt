SET(HEADER_PATH ${proLandOceanTry_SOURCE_DIR}/PrecomputedAtmosphericScattering)
SET(PROJECTNAME PrecomputedAtmosphericScattering)

INCLUDE_DIRECTORIES(
   ${HEADER_PATH}
   ${anttweakbar_INCLUDE_DIR}
   ${tiff_INCLUDE_DIR}
)
link_directories(${glew_LIBRARY_PATH} ${glut_LIBRARY_PATH} ${TIFF_LIBRARY_PATH})
# Adds headers to the "Header Files" group in visual studio
SOURCE_GROUP( "Header Files" ${HEADER_PATH}/*.h )

ADD_EXECUTABLE(
  ${PROJECTNAME}
  Main.cpp
)
target_link_libraries(
    ${PROJECTNAME} 	
	${glew_LIBRARY}
	${GLUT_LIBRARY}
	${TIFF_LIBRARY}
)
install(
    TARGETS ${PROJECTNAME}
	RUNTIME DESTINATION ${EXECUTABLE_OUTPUT_PATH}
)







