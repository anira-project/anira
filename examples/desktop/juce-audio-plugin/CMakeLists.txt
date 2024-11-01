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

project (${PROJECT_NAME} VERSION 0.0.1)

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
        GIT_TAG 7.0.8
        GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(JUCE)

if(APPLE)
	set(FORMATS_TO_BUILD AU VST3 Standalone)
else()
	set(FORMATS_TO_BUILD VST3 Standalone)
endif()

# add_compile_definitions(MODEL_TO_USE=1) # steerable-nafx
add_compile_definitions(MODEL_TO_USE=2) # GuitarLSTM
# add_compile_definitions(MODEL_TO_USE=3) # stateful-LSTM
# add_compile_definitions(MODEL_TO_USE=4) # simple-gain
# add_compile_definitions(MODEL_TO_USE=5) # simple-stereo-gain

set (TARGET_NAME anira-juce-plugin-example)

juce_add_plugin(${TARGET_NAME}
		# VERSION ...                               # Set this if the plugin version is different to the project version
		# ICON_BIG ...                              # ICON_* arguments specify a path to an image file to use as an icon for the Standalone
		# ICON_SMALL ...
		COMPANY_NAME "AckvaSchulz"
		# IS_SYNTH TRUE/FALSE                       # Is this a synth or an effect?
		# NEEDS_MIDI_INPUT TRUE/FALSE               # Does the plugin need midi input?
		# NEEDS_MIDI_OUTPUT TRUE/FALSE              # Does the plugin need midi output?
		# IS_MIDI_EFFECT TRUE/FALSE                 # Is this plugin a MIDI effect?
		# EDITOR_WANTS_KEYBOARD_FOCUS TRUE/FALSE    # Does the editor need keyboard focus?
		# COPY_PLUGIN_AFTER_BUILD TRUE/FALSE        # Should the plugin be installed to a default location after building?
		PLUGIN_MANUFACTURER_CODE AcSc              # A four-character manufacturer id with at least one upper-case character
		PLUGIN_CODE Anir                            # A unique four-character plugin id with exactly one upper-case character
		# GarageBand 10.3 requires the first letter to be upper-case, and the remaining letters to be lower-case

        VST3_AUTO_MANIFEST FALSE

        if(APPLE)
            HARDENED_RUNTIME_ENABLED TRUE
            HARDENED_RUNTIME_OPTIONS "com.apple.security.device.audio-input"
            MICROPHONE_PERMISSION_ENABLED TRUE
            MICROPHONE_PERMISSION_TEXT "Need access to your audio interface"
		endif()

		FORMATS ${FORMATS_TO_BUILD}                     # The formats to build. Other valid formats are: AAX Unity VST AU AUv3
		PRODUCT_NAME "anira-juce-plugin-example"			            # The name of the final executable, which can differ from the target name
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
    JUCE_DISPLAY_SPLASH_SCREEN=0
    DONT_SET_USING_JUCE_NAMESPACE=1

    # Backend-specific definitions
    $<$<BOOL:${BACKEND_LIBTORCH}>:USE_LIBTORCH>
    $<$<BOOL:${BACKEND_ONNXRUNTIME}>:USE_ONNXRUNTIME>
    $<$<BOOL:${BACKEND_TFLITE}>:USE_TFLITE>
)

target_link_libraries(${TARGET_NAME}
    PRIVATE
        anira::anira
        juce::juce_audio_utils
        juce::juce_dsp
        juce::juce_core
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags
)

if (MSVC)
	foreach(FORMAT IN LISTS FORMATS_TO_BUILD)
		string(CONCAT CONCATENATED_TARGET ${TARGET_NAME}_${FORMAT})
		add_custom_command(TARGET ${CONCATENATED_TARGET}
				POST_BUILD
				COMMAND ${CMAKE_COMMAND} -E copy_if_different
				${ANIRA_SHARED_LIBS_WIN}
				$<TARGET_FILE_DIR:${TARGET_NAME}>/${FORMAT})
	endforeach()
endif (MSVC)