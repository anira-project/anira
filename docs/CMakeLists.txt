# Define what content to include in the generated documentation
set(DOC_SOURCES
    # Main pages
    "${CMAKE_CURRENT_SOURCE_DIR}/sphinx/index.rst"
    "${CMAKE_CURRENT_SOURCE_DIR}/sphinx/usage.rst"
    "${CMAKE_CURRENT_SOURCE_DIR}/sphinx/benchmarking.rst"
    "${CMAKE_CURRENT_SOURCE_DIR}/sphinx/examples.rst"
)

# look for Doxygen package
# Needs to be installed i.e. sudo dnf install doxygen
find_package(Doxygen REQUIRED)

if (DOXYGEN_FOUND)
    # set input and output files
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/doxygen/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/doxygen/Doxyfile)

    # request to configure the file
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
    message(STATUS "Doxygen configuration generated")

    # Note: do not put "ALL" - this builds docs together with application EVERY TIME!
    add_custom_target(doxygen-docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM )
else (DOXYGEN_FOUND)
    message(WARNING "Doxygen need to be installed to generate the doxygen documentation")
endif (DOXYGEN_FOUND)

# This needs sphinx, furo and breathe to be installed i.e. sudo dnf install python3-sphinx python3-breathe python3-furo
find_program(SPHINX_EXECUTABLE
    NAMES sphinx-build
    DOC "Path to sphinx-build executable")
    
    
if (SPHINX_EXECUTABLE)
    set(SPHINX_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx)
    set(SPHINX_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/sphinx)

    configure_file(${SPHINX_SOURCE_DIR}/conf.py.in ${SPHINX_BUILD_DIR}/conf.py @ONLY)
    
    add_custom_target(sphinx-docs
    COMMAND ${SPHINX_EXECUTABLE} -b html 
    -c ${SPHINX_BUILD_DIR} 
    ${SPHINX_SOURCE_DIR} 
    ${SPHINX_BUILD_DIR}/html
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating HTML documentation with Sphinx"
    DEPENDS doxygen-docs ${DOC_SOURCES}
    VERBATIM )
else (SPHINX_EXECUTABLE)
    message(WARNING "Sphinx need to be installed to generate the sphinx documentation")
endif (SPHINX_EXECUTABLE)

find_program(BREATHE_EXECUTABLE
    NAMES breathe-apidoc
    DOC "Path to breathe-apidoc executable")

if (BREATHE_EXECUTABLE)
    add_custom_target(breathe-apidocs
        COMMAND ${BREATHE_EXECUTABLE} -o ${SPHINX_SOURCE_DIR}/api
            ${CMAKE_CURRENT_BINARY_DIR}/doxygen/xml
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Breathe"
        DEPENDS doxygen-docs
        VERBATIM )
else (BREATHE_EXECUTABLE)
    message(WARNING "Breathe need to be installed to generate the breathe documentation")
endif (BREATHE_EXECUTABLE)

add_custom_target(docs
    COMMENT "Build both Doxygen and Sphinx documentation"
    DEPENDS sphinx-docs
)

if (ANIRA_WITH_INSTALL)
    # Install documentation
    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/sphinx/html/
        DESTINATION share/doc/anira
        COMPONENT documentation
        OPTIONAL
    )
endif()