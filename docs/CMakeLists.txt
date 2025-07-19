# look for Doxygen package
find_package(Doxygen)

if (DOXYGEN_FOUND)
    # set input and output files
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/doxygen/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/doxygen/Doxyfile.out)
    set(PROJECT_INCLUDE_DIR "include")

    # request to configure the file
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
    message("Doxygen build started")

    # Note: do not put "ALL" - this builds docs together with application EVERY TIME!
    add_custom_target(doxygen-docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM )
else (DOXYGEN_FOUND)
  message("Doxygen need to be installed to generate the doxygen documentation")
endif (DOXYGEN_FOUND)

find_program(SPHINX_EXECUTABLE
    NAMES sphinx-build
    DOC "Path to sphinx-build executable")

if (SPHINX_EXECUTABLE)
    set(SPHINX_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx)
    set(SPHINX_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/sphinx)

    configure_file(${SPHINX_SOURCE_DIR}/conf.py.in ${SPHINX_BUILD_DIR}/conf.py @ONLY)

    add_custom_target(sphinx-docs
        COMMAND ${SPHINX_EXECUTABLE} -b html -c ${SPHINX_BUILD_DIR} ${SPHINX_SOURCE_DIR} ${SPHINX_BUILD_DIR}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating HTML documentation with Sphinx"
        VERBATIM )
else (SPHINX_EXECUTABLE)
  message("Sphinx need to be installed to generate the sphinx documentation")
endif (SPHINX_EXECUTABLE)

add_custom_target(docs
    COMMENT "Build both Doxygen and Sphinx documentation"
    DEPENDS doxygen-docs sphinx-docs
)