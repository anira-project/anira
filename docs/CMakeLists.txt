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
        COMMAND ${DOCS_VENV_DIR}/bin/pip install --quiet sphinx breathe shibuya myst-parser sphinx-js
        # Patch sphinx-js bug: redirectPrivateAliases.ts crashes on missing-export
        # symbols whose `declarations` is undefined (occurs with TypeDoc >= 0.27).
        # Replace `name.declarations![0]` with a guarded read. See pyodide/sphinx-js.
        COMMAND bash -c "perl -i -pe 's{const decl = name\\.declarations!\\[0\\];}{const decl = name.declarations?.[0]; if (!decl) continue;}' ${DOCS_VENV_DIR}/lib/python*/site-packages/sphinx_js/js/redirectPrivateAliases.ts"
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
    # Add :allow-dot-graphs: directive to all class and struct files.
    # Using perl -i for portability (BSD sed on macOS uses different -i syntax than GNU sed).
    COMMAND bash -c "find ${SPHINX_SOURCE_DIR}/api/class -name '*.rst' -exec perl -i -pe 's{^(\\.\\.\\s+doxygenclass::.*\\n)}{$1    :allow-dot-graphs:\\n}' {} +"
    COMMAND bash -c "find ${SPHINX_SOURCE_DIR}/api/struct -name '*.rst' -exec perl -i -pe 's{^(\\.\\.\\s+doxygenstruct::.*\\n)}{$1    :allow-dot-graphs:\\n}' {} +"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}

    COMMENT "Generating API documentation with Breathe using virtual environment"
    DEPENDS create_docs_venv doxygen-docs
    VERBATIM
)

# --- TypeDoc: generate JSON for the Anira Web TypeScript API -----------------
# sphinx-js consumes this JSON via the js: domain, producing rendering that
# matches the C++ side (Sphinx domain directives styled by the shibuya theme).
set(TYPEDOC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/typedoc)
set(TYPEDOC_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/typedoc)
set(TYPEDOC_JSON ${TYPEDOC_BUILD_DIR}/typedoc.json)
set(TYPEDOC_NPM_STAMP ${TYPEDOC_BUILD_DIR}/.npm-installed)

find_program(NPM_EXECUTABLE npm)

if (NPM_EXECUTABLE)
    file(MAKE_DIRECTORY ${TYPEDOC_BUILD_DIR})

    add_custom_command(
        OUTPUT ${TYPEDOC_NPM_STAMP}
        COMMAND ${NPM_EXECUTABLE} install --prefix ${TYPEDOC_DIR}
        COMMAND ${CMAKE_COMMAND} -E touch ${TYPEDOC_NPM_STAMP}
        DEPENDS ${TYPEDOC_DIR}/package.json
        COMMENT "Installing TypeDoc into docs/typedoc/node_modules"
        VERBATIM
    )

    add_custom_command(
        OUTPUT ${TYPEDOC_JSON}
        COMMAND ${NPM_EXECUTABLE} exec --prefix ${TYPEDOC_DIR} -- typedoc
                --options ${TYPEDOC_DIR}/typedoc.json
                --json ${TYPEDOC_JSON}
        DEPENDS ${TYPEDOC_NPM_STAMP} ${TYPEDOC_DIR}/typedoc.json
        WORKING_DIRECTORY ${TYPEDOC_DIR}
        COMMENT "Generating TypeDoc JSON for Anira Web"
        VERBATIM
    )

    add_custom_target(typedoc-json DEPENDS ${TYPEDOC_JSON})
else()
    message(WARNING "npm not found - Anira Web API reference will be skipped. Install Node.js to enable it.")
    add_custom_target(typedoc-json
        COMMAND ${CMAKE_COMMAND} -E echo "Skipping TypeDoc: npm not found"
    )
endif()

# Configure the Sphinx configuration file
configure_file(${SPHINX_SOURCE_DIR}/conf.py.in ${SPHINX_BUILD_DIR}/conf.py @ONLY)

# Collect all files in SPHINX_SOURCE_DIR as dependencies
file(GLOB_RECURSE SPHINX_SOURCE_FILES "${SPHINX_SOURCE_DIR}/*")

# sphinx-js shells out to `typedoc`, so the locally-installed binary in the
# docs/typedoc node_modules must be on PATH when sphinx-build runs.
add_custom_target(sphinx-docs
    COMMAND ${CMAKE_COMMAND} -E env
            "PATH=${TYPEDOC_DIR}/node_modules/.bin:$ENV{PATH}"
            ${DOCS_VENV_DIR}/bin/sphinx-build -b html
            -c ${SPHINX_BUILD_DIR}
            ${SPHINX_SOURCE_DIR}
            ${SPHINX_BUILD_DIR}/html
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating HTML documentation with Sphinx using virtual environment"
    DEPENDS create_docs_venv doxygen-docs breathe-apidocs typedoc-json ${SPHINX_SOURCE_FILES}
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