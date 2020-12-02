# Copyright (c) 2020 Jakob Meng, <jakobmeng@web.de>
# vim:set fileformat=unix shiftwidth=4 softtabstop=4 expandtab:
# kate: end-of-line unix; space-indent on; indent-width 4; remove-trailing-spaces modified;
#
# Redistribution and use is allowed according to the terms of the BSD license.
# For details see the accompanying COPYING-CMAKE-SCRIPTS file.
#

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

#################### project settings ####################

project(edamer VERSION 2020.11.0.0)

include(FeatureSummary)

#################### options ####################

option(EDAMER_ENABLE_SCALAR_INT "Enable scalar type int." ON)
option(EDAMER_ENABLE_SCALAR_FLOAT "Enable scalar type float." ON)
option(EDAMER_ENABLE_SCALAR_DOUBLE "Enable scalar type double." ON)
option(EDAMER_ENABLE_SCALAR_COMPLEX_FLOAT "Enable scalar type complex float." OFF)
option(EDAMER_ENABLE_SCALAR_COMPLEX_DOUBLE "Enable scalar type complex double." OFF)
option(EDAMER_ENABLE_MATRIX_DISTRIBUTION_CIRC_CIRC "Enable matrix distribution [CIRC,CIRC]." ON)
option(EDAMER_ENABLE_MATRIX_DISTRIBUTION_MC_MR     "Enable matrix distribution [MC,MR]."     ON)
option(EDAMER_ENABLE_MATRIX_DISTRIBUTION_MC_STAR   "Enable matrix distribution [MC,STAR]."   ON)
option(EDAMER_ENABLE_MATRIX_DISTRIBUTION_MD_STAR   "Enable matrix distribution [MD,STAR]."   ON)
option(EDAMER_ENABLE_MATRIX_DISTRIBUTION_MR_MC     "Enable matrix distribution [MR,MC]."     ON)
option(EDAMER_ENABLE_MATRIX_DISTRIBUTION_MR_STAR   "Enable matrix distribution [MR,STAR]."   ON)
option(EDAMER_ENABLE_MATRIX_DISTRIBUTION_STAR_MC   "Enable matrix distribution [STAR,MC]."   ON)
option(EDAMER_ENABLE_MATRIX_DISTRIBUTION_STAR_MD   "Enable matrix distribution [STAR,MD]."   ON)
option(EDAMER_ENABLE_MATRIX_DISTRIBUTION_STAR_MR   "Enable matrix distribution [STAR,MR]."   ON)
option(EDAMER_ENABLE_MATRIX_DISTRIBUTION_STAR_STAR "Enable matrix distribution [STAR,STAR]." ON)
option(EDAMER_ENABLE_MATRIX_DISTRIBUTION_STAR_VC   "Enable matrix distribution [STAR,VC]."   ON)
option(EDAMER_ENABLE_MATRIX_DISTRIBUTION_STAR_VR   "Enable matrix distribution [STAR,VR]."   ON)
option(EDAMER_ENABLE_MATRIX_DISTRIBUTION_VC_STAR   "Enable matrix distribution [VC,STAR]."   ON)
option(EDAMER_ENABLE_MATRIX_DISTRIBUTION_VR_STAR   "Enable matrix distribution [VR,STAR]."   ON)
option(EDAMER_ENABLE_TESTS "Build unit tests." OFF)

#################### find all used packages ####################

# NOTE: Keep list of packages in sync with dependencies in *-cmake.in!

find_package(hbrs-cmake)
set_package_properties(hbrs-cmake PROPERTIES
    PURPOSE "Required for CMake modules."
    TYPE REQUIRED)

find_package(hbrs-mpl)
set_package_properties(hbrs-mpl PROPERTIES
    PURPOSE "Required for linear algebra algorithms."
    TYPE REQUIRED)

find_package(Boost 1.62 COMPONENTS unit_test_framework system thread serialization chrono program_options filesystem iostreams regex exception log)
set_package_properties(Boost PROPERTIES
    PURPOSE "Required for meta programming, unit tests and others."
    TYPE REQUIRED)

find_package(MPI)
set_package_properties(MPI PROPERTIES
    PURPOSE "Required for distributed linear algebra algortihms."
    TYPE REQUIRED)

find_package(Python3 COMPONENTS Interpreter Development)
set_package_properties(Python3 PROPERTIES
    PURPOSE "Required for installing the Python module."
    TYPE REQUIRED)

# pybind11's FAQ advises to avoid using find_package(PythonInterp) and find_package(PythonLibs) from CMake and
# rely on pybind11 in detecting Python version, but we need e.g. Python3_SITEARCH to define the install paths.
#
# Ref.: https://pybind11.readthedocs.io/en/stable/faq.html#inconsistent-detection-of-python-version-in-cmake-and-pybind11

find_package(pybind11)
set_package_properties(pybind11 PROPERTIES
    PURPOSE "Required for creating the Python bindings of C++ code."
    TYPE REQUIRED)

find_package(mpi4py)
set_package_properties(mpi4py PROPERTIES
    PURPOSE "Required for bridging mpi4py and C++ code."
    TYPE REQUIRED)

feature_summary(WHAT REQUIRED_PACKAGES_NOT_FOUND FATAL_ON_MISSING_REQUIRED_PACKAGES)

#################### source settings ####################

enable_language(CXX)
enable_language(Python3)

# Put the include dirs which are in the source or build tree before all other include dirs,
# so the headers in the sources are preferred over the already installed ones.
# Ref.: kdelibs/cmake/modules/KDE4Defaults.cmake
set(CMAKE_INCLUDE_DIRECTORIES_PROJECT_BEFORE ON)

set(CXX_TEMPLATE_DEPTH 250            CACHE STRING "Set the maximum instantiation depth for template classes to n (-ftemplate-depth=n).") # increased depth is required by operators.hpp
set(CXX_TEMPLATE_BACKTRACE_LIMIT 250  CACHE STRING "Set the maximum number of template instantiation notes for a single warning or error to n (-ftemplate-backtrace-limit=n).")
set(CXX_ERROR_LIMIT 0                 CACHE STRING "Limits the maximum number of error messages to n (-fmax-errors=n / -ferror-limit=n).")
set(CXX_CONSTEXPR_BACKTRACE_LIMIT 250 CACHE STRING "-fconstexpr-backtrace-limit=n")
set(CXX_MACRO_BACKTRACE_LIMIT 250     CACHE STRING "-fmacro-backtrace-limit=n")

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)

# "CMake will generate tests only if the enable_testing() command has been invoked." [1]
# "Note that ctest expects to find a test file in the build directory root. Therefore, this command should be in the 
#  source directory root." [2]
# References:
# [1] https://cmake.org/cmake/help/latest/command/add_test.html#command:add_test
# [2] https://cmake.org/cmake/help/latest/command/enable_testing.html
if(EDAMER_ENABLE_TESTS)
    enable_testing()
endif()

#################### build settings ####################

include(MaybeAddFlags)
maybe_add_c_flag  (EDAMER_HAS_C_WALL                         "-Wall")
maybe_add_cxx_flag(EDAMER_HAS_CXX_WALL                       "-Wall")
#maybe_add_c_flag  (EDAMER_HAS_C_WEXTRA                       "-Wextra")
#maybe_add_cxx_flag(EDAMER_HAS_CXX_WEXTRA                     "-Wextra")
maybe_add_c_flag  (EDAMER_HAS_C_WPEDANTIC                    "-Wpedantic")
maybe_add_cxx_flag(EDAMER_HAS_CXX_WPEDANTIC                  "-Wpedantic")
maybe_add_c_flag  (EDAMER_HAS_C_WERROR_RETURN_TYPE           "-Werror=return-type")
maybe_add_cxx_flag(EDAMER_HAS_CXX_WERROR_RETURN_TYPE         "-Werror=return-type")
maybe_add_c_flag  (EDAMER_HAS_C_PEDANTIC_ERRORS              "-pedantic-errors")
maybe_add_cxx_flag(EDAMER_HAS_CXX_PEDANTIC_ERRORS            "-pedantic-errors")

maybe_add_cxx_flag(EDAMER_HAS_CXX_FTEMPLATE_DEPTH            "-ftemplate-depth=${CXX_TEMPLATE_DEPTH}")
maybe_add_cxx_flag(EDAMER_HAS_CXX_FTEMPLATE_BACKTRACE_LIMIT  "-ftemplate-backtrace-limit=${CXX_TEMPLATE_BACKTRACE_LIMIT}")
maybe_add_cxx_flag(EDAMER_HAS_CXX_FCONSTEXPR_BACKTRACE_LIMIT "-fconstexpr-backtrace-limit=${CXX_CONSTEXPR_BACKTRACE_LIMIT}")
maybe_add_cxx_flag(EDAMER_HAS_CXX_FMACRO_BACKTRACE_LIMIT     "-fmacro-backtrace-limit=${CXX_MACRO_BACKTRACE_LIMIT}")

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    maybe_add_cxx_flag(EDAMER_HAS_CXX_FERROR_LIMIT           "-ferror-limit=${CXX_ERROR_LIMIT}")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    maybe_add_cxx_flag(EDAMER_HAS_CXX_FMAX_ERRORS            "-fmax-errors=${CXX_ERROR_LIMIT}")
endif()

if(NOT EDAMER_HAS_CXX_FTEMPLATE_DEPTH)
    message(FATAL_ERROR "Your C++ compiler ${CMAKE_CXX_COMPILER} does not support '-ftemplate-depth=${CXX_TEMPLATE_DEPTH}', use it at your own risk.")
endif()

if(NOT EDAMER_HAS_CXX_FTEMPLATE_BACKTRACE_LIMIT)
    message(WARNING "Your C++ compiler ${CMAKE_CXX_COMPILER} does not support '-ftemplate-backtrace-limit=${CXX_TEMPLATE_BACKTRACE_LIMIT}', use it at your own risk.")
endif()

if(NOT EDAMER_HAS_CXX_FCONSTEXPR_BACKTRACE_LIMIT)
    message(WARNING "Your C++ compiler ${CMAKE_CXX_COMPILER} does not support '-fconstexpr-backtrace-limit=${CXX_CONSTEXPR_BACKTRACE_LIMIT}', use it at your own risk.")
endif()

if(NOT EDAMER_HAS_CXX_FMACRO_BACKTRACE_LIMIT)
    message(WARNING "Your C++ compiler ${CMAKE_CXX_COMPILER} does not support '-fmacro-backtrace-limit=${CXX_MACRO_BACKTRACE_LIMIT}', use it at your own risk.")
endif()

include(CheckIPOSupported)
check_ipo_supported(RESULT EDAMER_HAS_IPO)
if(EDAMER_HAS_IPO)
	set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
endif()

#################### install settings ####################

# Get relative SITEARCH path
execute_process (COMMAND "${Python3_EXECUTABLE}" -c "\
import sys;\
from distutils import sysconfig;\
sys.stdout.write(\
sysconfig.get_python_lib(prefix='',plat_specific=True,standard_lib=False)\
)"
    RESULT_VARIABLE _Python3_SITEARCH_RESULT
    OUTPUT_VARIABLE _Python3_SITEARCH)

if(NOT _Python3_SITEARCH_RESULT EQUAL 0)
    message(FATAL_ERROR "Could not detect Python's third-party platform dependent installation directory.")
endif()

# Offer a choice of overriding the installation directories
set(EDAMER_INSTALL_PYTHON_SITEARCH "${_Python3_SITEARCH}"
    CACHE PATH "install dir for 3rd-party platform dependent Python modules")

#################### tests ####################

# CMake's target test does not build but only executes tests
# Ref.: https://stackoverflow.com/a/736838/6490710
if(EDAMER_ENABLE_TESTS)
   set(_TESTS_MAYBE_ALL ALL)
endif()

add_custom_target(tests ${_TESTS_MAYBE_ALL} COMMENT "Build all tests.")

include(DevTools)
function(edamer_add_test target)
    add_executable(${target} EXCLUDE_FROM_ALL "${ARGN}")
    target_link_libraries(${target} cpp ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
    target_include_directories(${target}
        SYSTEM
        PUBLIC ${Boost_INCLUDE_DIRS})

    target_compile_definitions(${target} PUBLIC ${MPI_CXX_COMPILE_DEFINITIONS} BOOST_LOG_DYN_LINK)

    add_test(${target} ${CMAKE_CURRENT_BINARY_DIR}/${target})
    add_dependencies(tests ${target})
endfunction()

function(edamer_add_pytest target)
    set(paths ${ARGN})
    # Ensure that paths are inside CMAKE_CURRENT_SOURCE_DIR, else
    # the "copy_if_different" command below does not work correctly.
    foreach(path IN LISTS paths)
        get_filename_component(abs_path "${path}" ABSOLUTE)
        get_filename_component(dir "${abs_path}" DIRECTORY)
        if(NOT dir STREQUAL CMAKE_CURRENT_SOURCE_DIR)
            message(FATAL_ERROR "Path '${path}' points outside source directory '${CMAKE_CURRENT_SOURCE_DIR}'")
        endif()
    endforeach()

    set(src_paths ${ARGN})
    list(TRANSFORM src_paths PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
    set(bin_paths ${ARGN})
    list(TRANSFORM bin_paths PREPEND "${CMAKE_CURRENT_BINARY_DIR}/")

    add_custom_target(${target}
        COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=$<TARGET_FILE_DIR:cpp>/.."
            ${Python3_EXECUTABLE} -B -m flake8 ${src_paths}
        # Copy sources to CMAKE_CURRENT_BINARY_DIR in order to be able to run tests from build tree
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${src_paths} "${CMAKE_CURRENT_BINARY_DIR}"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS cpp py3)

    # Test must be run from CMAKE_CURRENT_BINARY_DIR instead of CMAKE_CURRENT_SOURCE_DIR. Else Python will prepend
    # the "${PROJECT_SOURCE_DIR}/src" to the module search path (PYTHONPATH), which causes all tests to fail with
    # "ModuleNotFoundError: No module named 'edamer.cpp'" because Python finds and uses module 'edamer' inside the
    # source tree instead of the module inside the build tree.
    add_test(NAME ${target}
        COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=$<TARGET_FILE_DIR:cpp>/.."
            ${Python3_EXECUTABLE} -B -m pytest -o log_cli=true -o log_cli_level=debug
            ${bin_paths}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

    add_dependencies(tests ${target})
endfunction()

function(edamer_add_pytest_mpi target)
    set(paths ${ARGN})
    # Ensure that paths are inside CMAKE_CURRENT_SOURCE_DIR, else
    # the "copy_if_different" command below does not work correctly.
    foreach(path IN LISTS paths)
        get_filename_component(abs_path "${path}" ABSOLUTE)
        get_filename_component(dir "${abs_path}" DIRECTORY)
        if(NOT dir STREQUAL CMAKE_CURRENT_SOURCE_DIR)
            message(FATAL_ERROR "Path '${path}' points outside source directory '${CMAKE_CURRENT_SOURCE_DIR}'")
        endif()
    endforeach()

    set(src_paths ${ARGN})
    list(TRANSFORM src_paths PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
    set(bin_paths ${ARGN})
    list(TRANSFORM bin_paths PREPEND "${CMAKE_CURRENT_BINARY_DIR}/")

    add_custom_target(${target}
        COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=$<TARGET_FILE_DIR:cpp>/.."
            ${Python3_EXECUTABLE} -B -m flake8 ${src_paths}
        # Copy sources to CMAKE_CURRENT_BINARY_DIR in order to be able to run tests from build tree
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${src_paths} "${CMAKE_CURRENT_BINARY_DIR}"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS cpp py3)

    # Test must be run from CMAKE_CURRENT_BINARY_DIR instead of CMAKE_CURRENT_SOURCE_DIR. Else Python will prepend
    # the "${PROJECT_SOURCE_DIR}/src" to the module search path (PYTHONPATH), which causes all tests to fail with
    # "ModuleNotFoundError: No module named 'edamer.cpp'" because Python finds and uses module 'edamer' inside the
    # source tree instead of the module inside the build tree.
    add_test(NAME ${target}
        COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=$<TARGET_FILE_DIR:cpp>/.."
            ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS}
            ${Python3_EXECUTABLE} -B -m pytest -o log_cli=true -o log_cli_level=debug
            #${MPIEXEC_POSTFLAGS}
            ${bin_paths}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

    add_dependencies(tests ${target})
endfunction()

#################### list the subdirectories ####################

add_subdirectory(src)

#################### summary ####################

feature_summary(WHAT ALL FATAL_ON_MISSING_REQUIRED_PACKAGES)
