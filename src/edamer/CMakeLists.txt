# Copyright (c) 2020 Jakob Meng, <jakobmeng@web.de>
# vim:set fileformat=unix shiftwidth=4 softtabstop=4 expandtab:
# kate: end-of-line unix; space-indent on; indent-width 4; remove-trailing-spaces modified;
#
# Redistribution and use is allowed according to the terms of the BSD license.
# For details see the accompanying COPYING-CMAKE-SCRIPTS file.
#

#################### configure files ####################

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/config.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/config.hpp"
    @ONLY)

#################### build ####################

add_library(cpp MODULE)

set_target_properties(cpp PROPERTIES OUTPUT_NAME "cpp")

get_target_property(cpp_TARGET_TYPE cpp TYPE)
if (cpp_TARGET_TYPE STREQUAL "STATIC_LIBRARY")
    target_compile_definitions(cpp INTERFACE EDAMER_STATIC_DEFINE)
endif ()

set_target_properties(cpp PROPERTIES
    VERSION ${cpp_VERSION}
    SOVERSION 0
    INTERFACE_cpp_MAJOR_VERSION ${cpp_VERSION})

set_property(TARGET cpp
    APPEND PROPERTY COMPATIBLE_INTERFACE_STRING cpp_MAJOR_VERSION)

target_include_directories(cpp
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/src> # for export.hpp
)

target_include_directories(cpp
    SYSTEM
    PUBLIC ${MPI_CXX_INCLUDE_DIRS}
    PUBLIC ${Boost_INCLUDE_DIRS}
    PUBLIC ${VTK_INCLUDE_DIRS}
    PUBLIC ${mpi4py_INCLUDE_DIRS})

target_compile_definitions(cpp
    PUBLIC ${MPI_CXX_COMPILE_DEFINITIONS} BOOST_LOG_DYN_LINK)

set_target_properties(cpp PROPERTIES
    COMPILE_FLAGS "${MPI_CXX_COMPILE_OPTIONS}"
    LINK_FLAGS "${MPI_CXX_LINK_FLAGS}"
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}")

target_link_libraries(cpp
    PUBLIC
        ${MPI_CXX_LIBRARIES}
        ${Boost_LIBRARIES}
        hbrs-mpl::hbrs_mpl
    PRIVATE
        pybind11::module)

include(GenerateExportHeader)
generate_export_header(cpp
    BASE_NAME EDAMER
    EXPORT_FILE_NAME export.hpp)

target_sources(cpp PRIVATE
    main.cpp)

add_library(py3 OBJECT)
target_sources(py3 PRIVATE
    __init__.py)

# For unit tests it is necessary that the C++ library as well as the Python code is available in the build directory,
# hence we synchronizes all Python files in the source and build directories.

# Copy Python files from source tree to build directory on every build if they have been updated.
# NOTE: Target update_py_files is not necessary because target py3 will copy all required source files to the build tree
add_custom_target(update_py_files
    COMMAND find . -iname \"*.py\" | while IFS= read -r FILE \; do
		DIR=$$\(dirname \"$$FILE\"\) && { [ ! -d \"$$DIR\" ] && ${CMAKE_COMMAND} -E make_directory \"$$DIR\" || true\; } &&
		${CMAKE_COMMAND} -E copy_if_different \"$$FILE\" \"${CMAKE_CURRENT_BINARY_DIR}/$$FILE\" \; done
	WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Delete Python files in build directory if they have no counterpart in the source tree.
add_custom_target(clean_py_files
    COMMAND find . -iname \"*.py\" | while IFS= read -r FILE \; do
		[ ! -e \"$$CMAKE_CURRENT_SOURCE_DIR/$$FILE\" ] &&
			${CMAKE_COMMAND} -E remove \"${CMAKE_CURRENT_BINARY_DIR}/$$FILE\" \; done
	WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_dependencies(py3 clean_py_files)

#################### list the subdirectories ####################

add_subdirectory(detail)
add_subdirectory(dt)
add_subdirectory(fn)

#################### install files ####################

install(TARGETS cpp
    LIBRARY DESTINATION "${EDAMER_INSTALL_PYTHON_SITEARCH}/edamer")

# Install selected Python3 files only
# NOTE: A command like the following would install all *.py files, i.e. including unit tests, which is not what we want:
#install(DIRECTORY ./ DESTINATION "${EDAMER_INSTALL_PYTHON_SITEARCH}/edamer" FILES_MATCHING PATTERN "*.py")
get_target_property(py3_srcs py3 SOURCES)
foreach(py3_src IN LISTS py3_srcs)
    get_filename_component(py3_src_abs "${py3_src}" ABSOLUTE)
    file(RELATIVE_PATH py3_src_rel "${PROJECT_SOURCE_DIR}/src" "${py3_src_abs}")
    get_filename_component(py3_src_dir "${py3_src_rel}" DIRECTORY)
    install(FILES "${py3_src}" DESTINATION "${EDAMER_INSTALL_PYTHON_SITEARCH}/${py3_src_dir}")
endforeach()
