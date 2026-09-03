cmake_minimum_required(VERSION 3.15)

# Sets the minimum macOS version
if (APPLE)
	set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum version of the target platform" FORCE) 
	if(CMAKE_OSX_DEPLOYMENT_TARGET)
		message("The minimum macOS version is set to " $CACHE{CMAKE_OSX_DEPLOYMENT_TARGET}.)
	endif()
endif ()

cmake_minimum_required(VERSION 3.15)

set (PROJECT_NAME anira-juce-plugin-example)

project (${PROJECT_NAME} VERSION ${PROJECT_VERSION})

# Sets the cpp language minimum
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# ==============================================================================

# Plug-In / Standalone setup

# ==============================================================================

message(STATUS "Cloning JUCE...")
include(FetchContent)

FetchContent_Declare(JUCE
        GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
        GIT_TAG 8.0.14
        GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(JUCE)

if(APPLE)
	set(FORMATS_TO_BUILD AU VST3 Standalone)
else()
	set(FORMATS_TO_BUILD VST3 Standalone)
endif()

# The model the plugin runs. Every variant loads its model file and contract file from
# extras/models (model_files.h) the same way; see PluginProcessor.h.
# 0 -> steerable-nafx CNN
# 1 -> steerable-nafx CNN, with the configuration files and the model files compiled into the
#      plugin (juce_add_binary_data) instead of read from disk
# 2 -> GuitarLSTM
# 3 -> stateful LSTM
# 4 -> SimpleGainNetwork, mono
# 5 -> SimpleGainNetwork, stereo
# 6 -> RAVE funk drum
# 7 -> RAVE funk drum as an encoder and a decoder handler
set(MODEL_TO_USE 1 CACHE STRING "The model the example plugin runs, 0 to 7; see PluginProcessor.h")
add_compile_definitions(MODEL_TO_USE=${MODEL_TO_USE})

set (TARGET_NAME anira-juce-plugin-example)

juce_add_plugin(${TARGET_NAME}
		# VERSION ...                               # Set this if the plugin version is different to the project version
		# ICON_BIG ...                              # ICON_* arguments specify a path to an image file to use as an icon for the Standalone
		# ICON_SMALL ...
		COMPANY_NAME "AniraProject"
		# IS_SYNTH TRUE/FALSE                       # Is this a synth or an effect?
		# NEEDS_MIDI_INPUT TRUE/FALSE               # Does the plugin need midi input?
		# NEEDS_MIDI_OUTPUT TRUE/FALSE              # Does the plugin need midi output?
		# IS_MIDI_EFFECT TRUE/FALSE                 # Is this plugin a MIDI effect?
		# EDITOR_WANTS_KEYBOARD_FOCUS TRUE/FALSE    # Does the editor need keyboard focus?
		# COPY_PLUGIN_AFTER_BUILD TRUE/FALSE        # Should the plugin be installed to a default location after building?
		PLUGIN_MANUFACTURER_CODE Anir              # A four-character manufacturer id with at least one upper-case character
		PLUGIN_CODE Anir                            # A unique four-character plugin id with exactly one upper-case character
		# GarageBand 10.3 requires the first letter to be upper-case, and the remaining letters to be lower-case

        VST3_AUTO_MANIFEST TRUE

        if(APPLE)
            HARDENED_RUNTIME_ENABLED TRUE
            HARDENED_RUNTIME_OPTIONS "com.apple.security.device.audio-input"
            MICROPHONE_PERMISSION_ENABLED TRUE
            MICROPHONE_PERMISSION_TEXT "Need access to your audio interface"
		endif()

		FORMATS ${FORMATS_TO_BUILD}                     # The formats to build. Other valid formats are: AAX Unity VST AU AUv3
		PRODUCT_NAME "Anira JUCE Example"			            # The name of the final executable, which can differ from the target name
		)

juce_generate_juce_header(${TARGET_NAME})

# Add all source files to file list
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/*.h)

# Add all sources to target
target_sources(${TARGET_NAME} PRIVATE ${SOURCES})

# Add include directories for all folders in the source
file(GLOB_RECURSE SOURCE_DIRS LIST_DIRECTORIES true ${CMAKE_CURRENT_LIST_DIR}/*)
list(APPEND SOURCE_DIRS ${CMAKE_CURRENT_LIST_DIR})

foreach (DIR ${SOURCE_DIRS})
    if (IS_DIRECTORY ${DIR})
        target_include_directories(${TARGET_NAME} PRIVATE ${DIR})
    endif ()
endforeach ()

target_compile_definitions(${TARGET_NAME}
    PRIVATE
    # JUCE_WEB_BROWSER and JUCE_USE_CURL would be on by default, but you might not need them.
    JUCE_WEB_BROWSER=0  # If you remove this, add `NEEDS_WEB_BROWSER TRUE` to the `juce_add_plugin` call
    JUCE_USE_CURL=0     # If you remove this, add `NEEDS_CURL TRUE` to the `juce_add_plugin` call
    JUCE_VST3_CAN_REPLACE_VST2=0

    # Backend-specific definitions
    $<$<BOOL:${BACKEND_LIBTORCH}>:USE_LIBTORCH>
    $<$<BOOL:${BACKEND_ONNXRUNTIME}>:USE_ONNXRUNTIME>
    $<$<BOOL:${BACKEND_TFLITE}>:USE_TFLITE>
)

if(UNIX AND NOT APPLE)
	target_compile_definitions(${TARGET_NAME} PRIVATE JUCE_JACK=1)
endif()

target_link_libraries(${TARGET_NAME}
    PRIVATE
        anira::anira
        juce::juce_audio_utils
        juce::juce_dsp
        juce::juce_core
    PUBLIC
        juce::juce_recommended_config_flags
        # No juce_recommended_lto_flags: JUCE's LTO breaks the link under
        # clang + GNU ld (undefined vtable references), and an example gains
        # nothing from LTO.
        juce::juce_recommended_warning_flags
)

if(MODEL_TO_USE EQUAL 1)
	# Variant 1 ships without the model tree: the CNN's model file, its contract file and the
	# four exports the model file names are compiled into the plugin. The symbol names are
	# JUCE's (the file name without punctuation, the extension after an underscore).
	set(_cnn_dir "${ANIRA_EXTRAS_MODELS_DIR}/cnn")
	juce_add_binary_data(BinaryData
			HEADER_NAME "BinaryData.h"
			NAMESPACE "BinaryData"
			SOURCES
			${_cnn_dir}/cnn.model.json
			${_cnn_dir}/cnn.contract.json
			${_cnn_dir}/steerable-nafx/models/model_0/steerable-nafx-dynamic.pt
			${_cnn_dir}/steerable-nafx/models/model_0/steerable-nafx-libtorch-dynamic.onnx
			${_cnn_dir}/steerable-nafx/models/model_0/steerable-nafx-dynamic.tflite
			${_cnn_dir}/steerable-nafx/models/model_0/steerable-nafx-executorch.pte
	)

	target_link_libraries(${TARGET_NAME}
			PRIVATE
			BinaryData
	)
endif()

# Only copy runtime DLLs when there are any (a fully static build has none).
if (MSVC AND ANIRA_SHARED_LIBS_WIN)
	foreach(FORMAT IN LISTS FORMATS_TO_BUILD)
		string(CONCAT CONCATENATED_TARGET ${TARGET_NAME}_${FORMAT})
		# The format target's own directory — generator-agnostic (the base
		# target's dir plus /<FORMAT> only matched the VS generator's layout).
		add_custom_command(TARGET ${CONCATENATED_TARGET}
				POST_BUILD
				COMMAND ${CMAKE_COMMAND} -E copy_if_different
				${ANIRA_SHARED_LIBS_WIN}
				$<TARGET_FILE_DIR:${CONCATENATED_TARGET}>
		)
	endforeach()

	# The VS generator puts JUCE's helper tools in a per-config subdirectory;
	# single-config generators (Ninja) do not.
	if(CMAKE_GENERATOR MATCHES "Visual Studio")
		set(JUCE_MANIFEST_GENERATOR_PATH "${CMAKE_BINARY_DIR}/examples/juce-audio-plugin/${CMAKE_BUILD_TYPE}")
	else()
		set(JUCE_MANIFEST_GENERATOR_PATH "${CMAKE_BINARY_DIR}/examples/juce-audio-plugin")
	endif()

	add_custom_command(TARGET ${TARGET_NAME}_VST3
			PRE_BUILD
			COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${ANIRA_SHARED_LIBS_WIN}
			${JUCE_MANIFEST_GENERATOR_PATH}
	)
endif ()

if(ANIRA_WITH_INSTALL)
    include(${CMAKE_CURRENT_LIST_DIR}/install.cmake)
endif()
