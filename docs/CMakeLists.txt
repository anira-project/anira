# Set up Python virtual environment for Sphinx
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
set(DOCS_VENV_DIR "${CMAKE_CURRENT_BINARY_DIR}/venv")

# Use a more robust approach with a custom target for venv creation
add_custom_target(create_docs_venv)

# Check if venv already exists
if(NOT EXISTS ${DOCS_VENV_DIR})
    add_custom_command(
        TARGET create_docs_venv
        POST_BUILD
        COMMAND ${Python3_EXECUTABLE} -m venv ${DOCS_VENV_DIR}
        COMMAND ${DOCS_VENV_DIR}/bin/pip install --upgrade pip
        COMMAND ${DOCS_VENV_DIR}/bin/pip install --quiet sphinx breathe shibuya
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Creating virtual environment for Sphinx documentation"
        VERBATIM
    )
endif()

# look for Doxygen package
# Needs to be installed i.e. sudo dnf install doxygen graphviz
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

# Set up directories for Sphinx documentation
set(SPHINX_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx)
set(SPHINX_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/sphinx)

# Use breathe-apidoc from the virtual environment
add_custom_target(breathe-apidocs
    COMMAND ${DOCS_VENV_DIR}/bin/breathe-apidoc -o ${SPHINX_SOURCE_DIR}/api
    ${CMAKE_CURRENT_BINARY_DIR}/doxygen/xml
    -g class,struct -f
    COMMAND ${CMAKE_COMMAND} -E remove -f ${SPHINX_SOURCE_DIR}/api/struct/structanira_1_1InferenceConfig_1_1Defaults.rst
    COMMAND ${CMAKE_COMMAND} -E remove -f ${SPHINX_SOURCE_DIR}/api/struct/structanira_1_1SessionElement_1_1ThreadSafeStruct.rst
    # Add :allow-dot-graphs: directive to all class and struct files
    COMMAND bash -c "find ${SPHINX_SOURCE_DIR}/api/class -name '*.rst' -exec sed -i '/^\\.\\.\\s\\+doxygenclass::/a\\    :allow-dot-graphs:' {} +"
    COMMAND bash -c "find ${SPHINX_SOURCE_DIR}/api/struct -name '*.rst' -exec sed -i '/^\\.\\.\\s\\+doxygenstruct::/a\\    :allow-dot-graphs:' {} +"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}

    COMMENT "Generating API documentation with Breathe using virtual environment"
    DEPENDS create_docs_venv doxygen-docs
    VERBATIM
)

# Configure the Sphinx configuration file
configure_file(${SPHINX_SOURCE_DIR}/conf.py.in ${SPHINX_BUILD_DIR}/conf.py @ONLY)

# Collect all files in SPHINX_SOURCE_DIR as dependencies
file(GLOB_RECURSE SPHINX_SOURCE_FILES "${SPHINX_SOURCE_DIR}/*")

add_custom_target(sphinx-docs
    COMMAND ${DOCS_VENV_DIR}/bin/sphinx-build -b html 
    -c ${SPHINX_BUILD_DIR} 
    ${SPHINX_SOURCE_DIR} 
    ${SPHINX_BUILD_DIR}/html
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating HTML documentation with Sphinx using virtual environment"
    DEPENDS create_docs_venv doxygen-docs breathe-apidocs ${SPHINX_SOURCE_FILES}
    VERBATIM
)

if (ANIRA_WITH_INSTALL)
    # Install documentation
    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/sphinx/html/
        DESTINATION share/doc/anira
        COMPONENT documentation
        OPTIONAL
    )
endif()