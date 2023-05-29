cmake_minimum_required (VERSION 3.0.2)
project(astro LANGUAGES C CXX CUDA)

find_package(OpenGL REQUIRED)
find_package(CUDA REQUIRED)

set (VERBOSE ON)
set (CMAKE_MESSAGE_LOG_LEVEL VERBOSE)

#### Directory definition ####
set (PROJECT_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/include")
set (PROJECT_SRC_DIR      "${PROJECT_SOURCE_DIR}/src")
set (PROJECT_LIBS_DIR     "${PROJECT_SOURCE_DIR}/libs")

#### Source Files ####

# Gets the list of files inside the src directory
file(GLOB_RECURSE SOURCES "${PROJECT_SRC_DIR}/glad.c" "${PROJECT_SRC_DIR}/*.cpp")
file(GLOB_RECURSE SOURCES_CUDA "${PROJECT_SRC_DIR}/*.cu")

#### Header Files ####

# -- CUDA --
include_directories(${CUDA_INCLUDE_DIRS})

# -- Project Includes --
include_directories("${PROJECT_INCLUDE_DIRS}")
cuda_include_directories("${PROJECT_INCLUDE_DIRS}")

#### External Libraries ####
set( GLFW_SHARED_LIB "${PROJECT_LIBS_DIR}/glfw3.lib")

# Compilation Flags
set(COMPILE_FLAGS "")
set(LINKER_FLAGS  "")

if (${CMAKE_CXX_COMPILER_ID} STREQUAL Clang)
    set(COMPILE_FLAGS "${COMPILE_FLAGS} -W -Wall -pedantic -std=c++11 -Wshadow -O2 -g")
    set(LINKER_FLAGS  "${LINKER_FLAGS} -lm")
elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL GNU)
    set(COMPILE_FLAGS "${COMPILE_FLAGS} -W -Wall -pedantic -std=c++11 -Wshadow -O2 -g")
    set(LINKER_FLAGS  "${LINKER_FLAGS} -lm")
elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL Intel)
  # using Intel Compiler
elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
    set(COMPILE_FLAGS "${COMPILE_FLAGS}")
    set(LINKER_FLAGS  "${LINKER_FLAGS}")
endif()

# Cmake compilation flags redefinition
set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} ${COMPILE_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${LINKER_FLAGS}")

set(cuda_additional_flags "-arch=native")

# findCUDA Options
set(CUDA_VERBOSE_BUILD OFF)
set(CUDA_SEPARABLE_COMPILATION ON) # Allow multiple CUDA files compilation

# Generating the commands to compile .cu into .ptx
CUDA_WRAP_SRCS(astro PTX CUDA_PTX_FILES ${SOURCES_CUDA} OPTIONS ${cuda_additional_flags})

cuda_add_executable(astro ${SOURCES} ${SOURCES_CUDA} ${CUDA_PTX_FILES} OPTIONS ${cuda_additional_flags})

target_link_libraries(astro ${GLFW_SHARED_LIB}  ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_curand_LIBRARY})
