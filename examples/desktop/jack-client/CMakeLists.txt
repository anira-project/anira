cmake_minimum_required(VERSION 3.15)

# Sets the minimum macOS version
if (APPLE)
	set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum version of the target platform" FORCE) 
	if(CMAKE_OSX_DEPLOYMENT_TARGET)
		message("The minimum macOS version is set to " $CACHE{CMAKE_OSX_DEPLOYMENT_TARGET}.)
	endif()
endif ()

set (PROJECT_NAME anira-jack-client-example)

project (${PROJECT_NAME} VERSION 0.0.1)

# Sets the cpp language minimum
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# add_compile_definitions(MODEL_TO_USE=1) # steerable-nafx
add_compile_definitions(MODEL_TO_USE=2) # GuitarLSTM
# add_compile_definitions(MODEL_TO_USE=3) # stateful-LSTM

add_compile_definitions(VERSION="${PROJECT_VERSION}")

add_executable(${PROJECT_NAME} main.cpp JackClient.cpp)
target_link_libraries(${PROJECT_NAME} anira::anira jack)