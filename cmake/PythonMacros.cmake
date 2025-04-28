# Python macros
# ~~~~~~~~~~~~~
# This file defines the following macros:
#
# python_compile(LIST_OF_SOURCE_FILES)
#     Byte compile the py force files listed in the LIST_OF_SOURCE_FILES.
#     Compiled pyc files are stored in PYTHON_COMPILED_FILES, corresponding py
#     files are stored in PYTHON_COMPILE_PY_FILES
#
# python_install_all(DESINATION_DIR LIST_OF_SOURCE_FILES)
#     Install @LIST_OF_SOURCE_FILES, which is a list of Python .py files,
#     into the destination directory during install. The file will be byte
#     compiled and both the .py file and .pyc file will be installed.
#
# python_install_module(MODULE_NAME LIST_OF_SOURCE_FILES)
#     Similiar to #python_install_all(), but the files are automatically
#     installed to the site-package directory of python as module MODULE_NAME.

#   Copyright (C) 2012~2012 by Yichao Yu
#   yyc1992@gmail.com
#
#   This program is free software; you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, version 2 of the License.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program; if not, write to the
#   Free Software Foundation, Inc.,
#   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.

# This file incorporates work covered by the following copyright and
# permission notice:
#
#     Copyright (c) 2007, Simon Edwards <simon@simonzone.com>
#     Redistribution and use is allowed according to the terms of the BSD
#     license. For details see the accompanying COPYING-CMAKE-SCRIPTS file.
#

include(CMakeVarMacros)
include(CMakePathMacros)

get_filename_component(_py_cmake_module_dir
  "${CMAKE_CURRENT_LIST_FILE}" PATH)
set(_cmake_python_helper "${_py_cmake_module_dir}/cmake-python-helper.py")
if(NOT EXISTS "${_cmake_python_helper}")
  message(FATAL_ERROR "The file cmake-python-helper.py does not exist in ${_py_cmake_module_dir} (the directory where PythonMacros.cmake is located). Check your installation.")
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)

execute_process(COMMAND ${Python_EXECUTABLE}
  "${_cmake_python_helper}" --get-sys-info OUTPUT_VARIABLE python_config)
if(python_config)
  string(REGEX REPLACE ".*exec_prefix:([^\n]+).*$" "\\1"
    PYTHON_PREFIX ${python_config})
  if(CYGWIN)
    # We want to do this translation on the msys2 cmake (/usr/bin/cmake)
    # since it would otherwise mess up the installation path.
    # OTOH, we don't want to do this on the mingw cmake since
    # passing it a unix path seems to cause it to install it under C: followed by the path
    # specified. (i.e. `/c/abcd` turns into `C:/c/abcd` even though `/c/abcd`
    # is supposed to be `C:/abcd`)
    # I'm not really sure what are all the cmake versions out there on windows but at least
    # CYGWIN is set on the msys2 cmake but not the mingw one so
    # this is what we use to distinguish the two for now...
    cmake_utils_cygpath(PYTHON_PREFIX "${PYTHON_PREFIX}")
  endif()
  string(REGEX REPLACE ".*\nmagic_tag:([^\n]*).*$" "\\1"
    PYTHON_MAGIC_TAG ${python_config})
  string(REGEX REPLACE ".*\nextension_suffix:([^\n]*).*$" "\\1"
    PYTHON_EXTENSION_SUFFIX ${python_config})
endif()

find_program(CYTHON_PATH NAMES cython cython3)
if(CYTHON_PATH)
  execute_process(COMMAND "${CYTHON_PATH}" --version OUTPUT_VARIABLE cython_config)
  string(REGEX REPLACE ".*version ([.0-9]*).*$" "\\1" CYTHON_VERSION "${cython_config}")
  if("${CYTHON_VERSION}" STREQUAL "")
    set(CYTHON_FOUND False CACHE INTERNAL "Cython status" FORCE)
  else()
    set(CYTHON_FOUND True CACHE INTERNAL "Cython status" FORCE)
  endif()
else()
  set(CYTHON_FOUND False CACHE INTERNAL "Cython status" FORCE)
endif()

option(ENABLE_CYTHON "Whether cython should be enabled if detected" On)
if(DEFINED PYTHON_SITE_INSTALL_DIR)
  # If the user supplied the PYTHON_SITE_INSTALL_DIR option we want to
  # keep using it across reconfigurations so PYTHON_SITE_INSTALL_DIR should be cached.
  # However, if the user didn't supply one and we are using the default
  # I'd like it to be updating if the python version changed automatically
  # on a reconfigure so the variable we use for installation shouldn't be cached.
  # Hence we assign the user option or the find_package(Python) value
  # to a non-cached variable.
  set(_PYTHON_SITE_INSTALL_DIR "${PYTHON_SITE_INSTALL_DIR}")
else()
  set(_PYTHON_SITE_INSTALL_DIR "${Python_SITEARCH}")
endif()
option(ENABLE_CYTHON_COVERAGE "Whether cython coverage should be enabled" Off)

function(require_cython ver)
  if(CYTHON_FOUND AND "${CYTHON_VERSION}" VERSION_LESS "${ver}")
    set(CYTHON_FOUND False CACHE INTERNAL "Cython status" FORCE)
  endif()
endfunction()

set(CYTHON_DEF_LANG "C" CACHE INTERNAL "Cython default language" FORCE)
function(cython_default_language lang)
  set(CYTHON_DEF_LANG "${lang}" CACHE INTERNAL "Cython default language" FORCE)
endfunction()

add_custom_target(cython-all-pxd-target)
add_custom_target(all-python-target DEPENDS cython-all-pxd-target)

function(_python_compile SOURCE_FILE OUT_PY OUT_PYC OUT_TGT)
  # Filename
  get_filename_component(src_base "${SOURCE_FILE}" NAME_WE)

  cmake_utils_abs_path(src "${SOURCE_FILE}")
  get_filename_component(src_path "${src}" PATH)
  cmake_utils_is_subpath(issub "${CMAKE_BINARY_DIR}" "${src}")
  if(issub)
    # Already in the bin dir
    # Don't copy the file onto itself.
    set(dst "${src}")
    set(dst_path "${src_path}")
    cmake_utils_src_to_bin(basedir "${_BASE_DIR}")
  else()
    cmake_utils_src_to_bin(dst "${src}")
    cmake_utils_is_subpath(issub "${CMAKE_BINARY_DIR}" "${dst}")
    if(NOT issub)
      message(FATAL_ERROR "Cannot determine binary directory for ${src}")
    endif()
    get_filename_component(dst_path "${dst}" PATH)
    file(MAKE_DIRECTORY "${dst_path}")
    add_custom_command(
      OUTPUT "${dst}"
      COMMAND ${CMAKE_COMMAND} -E copy "${src}" "${dst}"
      DEPENDS "${src}")
    set(basedir "${_BASE_DIR}")
  endif()
  file(RELATIVE_PATH rel_path "${basedir}" "${src_path}/${src_base}")
  string(REGEX REPLACE "[^-._a-z0-9A-Z]" "_" target_name "${rel_path}")
  cmake_utils_get_unique_target(python-${target_name} _py_compile_target)

  # PEP 3147
  set(dst_pyc "${dst_path}/__pycache__/${src_base}.${PYTHON_MAGIC_TAG}.pyc")
  # should be fine, just in case
  file(MAKE_DIRECTORY "${dst_path}/__pycache__")
  add_custom_command(
    OUTPUT "${dst_pyc}"
    COMMAND ${Python_EXECUTABLE} "${_cmake_python_helper}" --compile "${dst}"
    DEPENDS "${dst}")

  add_custom_target("${_py_compile_target}" ALL DEPENDS "${dst}" "${dst_pyc}")
  add_dependencies(all-python-target "${_py_compile_target}")
  set(${OUT_PY} "${dst}" PARENT_SCOPE)
  set(${OUT_PYC} "${dst_pyc}" PARENT_SCOPE)
  set(${OUT_TGT} "${_py_compile_target}" PARENT_SCOPE)
endfunction()

define_property(DIRECTORY PROPERTY CYTHON_INCLUDE_DIRECTORIES INHERITED
  FULL_DOCS "Cython include paths"
  BRIEF_DOCS "Cython include paths")
define_property(SOURCE PROPERTY CYTHON_INCLUDE_DIRECTORIES INHERITED
  FULL_DOCS "Cython include paths"
  BRIEF_DOCS "Cython include paths")

define_property(DIRECTORY PROPERTY CYTHON_LINK_LIBRARIES INHERITED
  FULL_DOCS "Cython link libraries"
  BRIEF_DOCS "Cython link libraries")
define_property(SOURCE PROPERTY CYTHON_LINK_LIBRARIES INHERITED
  FULL_DOCS "Cython link libraries"
  BRIEF_DOCS "Cython link libraries")

function(cython_include_directories)
  set_property(DIRECTORY APPEND PROPERTY CYTHON_INCLUDE_DIRECTORIES ${ARGN})
endfunction()
function(cython_link_libraries)
  set_property(DIRECTORY APPEND PROPERTY CYTHON_LINK_LIBRARIES ${ARGN})
endfunction()

function(_cython_compile SOURCE_FILE OUT_PYX OUT_PXD OUT_C OUT_SO OUT_TGT)
  # Filename
  get_filename_component(src_base "${SOURCE_FILE}" NAME_WE)

  get_source_file_property(CY_LANG "${SOURCE_FILE}" CYTHON_LANGUAGE)
  if(NOT CY_LANG)
    set(CY_LANG "${CYTHON_DEF_LANG}")
  endif()
  if("${CY_LANG}" STREQUAL C)
    set(IS_CPP 0)
  elseif("${CY_LANG}" STREQUAL "C++" OR "${CY_LANG}" STREQUAL "CPP" OR "${CY_LANG}" STREQUAL "CXX")
    set(IS_CPP 1)
  else()
    message(FATAL_ERROR "Unknown cython language ${CY_LANG}")
  endif()

  cmake_utils_abs_path(src "${SOURCE_FILE}")
  get_filename_component(src_path "${src}" PATH)
  cmake_utils_is_subpath(issub "${CMAKE_BINARY_DIR}" "${src}")

  if(issub)
    set(dst "${src}")
    set(dst_path "${src_path}")
    set(need_copy 0)
    cmake_utils_src_to_bin(basedir "${_BASE_DIR}")
  else()
    cmake_utils_src_to_bin(dst "${src}")
    cmake_utils_is_subpath(issub "${CMAKE_BINARY_DIR}" "${dst}")
    if(NOT issub)
      message(FATAL_ERROR "Cannot determine binary directory for ${src}")
    endif()
    get_filename_component(dst_path "${dst}" PATH)
    file(MAKE_DIRECTORY "${dst_path}")
    set(need_copy 1)
    set(basedir "${_BASE_DIR}")
  endif()
  file(RELATIVE_PATH rel_path "${basedir}" "${src_path}/${src_base}")
  string(REGEX REPLACE "[^-._a-z0-9A-Z]" "_" target_name "${rel_path}")
  cmake_utils_get_unique_target(cython-${target_name} _cy_compile_target)

  if(IS_CPP)
    set(src_c "${src_path}/${src_base}.cpp")
    set(dst_c "${dst_path}/${src_base}.cpp")
  else()
    set(src_c "${src_path}/${src_base}.c")
    set(dst_c "${dst_path}/${src_base}.c")
  endif()

  # Copying the sources files over
  set(src_pxd "${src_path}/${src_base}.pxd")
  set(dst_pxd "${dst_path}/${src_base}.pxd")
  set(pyx_dep)
  if(need_copy)
    add_custom_command(
      OUTPUT "${dst}"
      COMMAND ${CMAKE_COMMAND} -E copy "${src}" "${dst}"
      DEPENDS "${src}")
    add_custom_target("${_cy_compile_target}-pyx"
      DEPENDS "${dst}")
    set(pyx_dep "${_cy_compile_target}-pyx")
    if(EXISTS "${src_pxd}")
      add_custom_command(
        OUTPUT "${dst_pxd}"
        COMMAND ${CMAKE_COMMAND} -E copy "${src_pxd}" "${dst_pxd}"
        DEPENDS "${src_pxd}")
      add_custom_target("${_cy_compile_target}-pxd"
        DEPENDS "${dst_pxd}")
      add_dependencies(cython-all-pxd-target "${_cy_compile_target}-pxd")
      set(pyx_dep ${pyx_dep} "${_cy_compile_target}-pxd" cython-all-pxd-target)
    endif()
  endif()
  if(CYTHON_FOUND AND ENABLE_CYTHON)
    if(IS_CPP)
      set(lang_args --cplus)
    else()
      set(lang_args)
    endif()
    if(ENABLE_CYTHON_COVERAGE)
      set(coverage_args -X linetrace=True)
    else()
      set(coverage_args)
    endif()
    get_property(cython_include_dirs SOURCE "${SOURCE_FILE}"
      PROPERTY CYTHON_INCLUDE_DIRECTORIES)
    set(include_args)
    foreach(dir ${cython_include_dirs})
      set(include_args ${include_args} -I "${dir}")
    endforeach()
    execute_process(COMMAND "${CMAKE_COMMAND}" -E touch "${dst_c}.dep2")
    add_custom_command(
      OUTPUT "${dst_c}"
      COMMAND "${CYTHON_PATH}" ${lang_args} ${include_args} -o "${dst_c}"
      -w "${basedir}" --depfile ${coverage_args} "${src}"
      COMMAND ${Python_EXECUTABLE} "${_py_cmake_module_dir}/fix-deps-file.py"
      "${dst_c}.dep" "${dst_c}.dep2" "${basedir}"
      DEPENDS "${src}" ${pyx_dep}
      DEPFILE "${dst_c}.dep2"
    )
    if(NOT TARGET cython-copyback)
      add_custom_target(cython-copyback)
    endif()
    add_custom_command(
      OUTPUT "${src_c}"
      COMMAND ${CMAKE_COMMAND} -E copy "${dst_c}" "${src_c}"
      DEPENDS "${dst_c}")
    add_custom_target("${_cy_compile_target}-c" ALL
      DEPENDS "${dst_c}")
    add_custom_target("${_cy_compile_target}-copyback"
      DEPENDS "${src_c}" "${_cy_compile_target}-c")
    add_dependencies(cython-copyback "${_cy_compile_target}-copyback")
  else()
    if(need_copy)
      add_custom_command(
        OUTPUT "${dst_c}"
        COMMAND ${CMAKE_COMMAND} -E copy "${src_c}" "${dst_c}"
        DEPENDS "${src_c}")
    endif()
    add_custom_target("${_cy_compile_target}-c"
      DEPENDS "${dst_c}")
  endif()

  set(dst_so_name "${src_base}${PYTHON_EXTENSION_SUFFIX}")
  set(dst_so "${dst_path}/${dst_so_name}")
  add_library("${_cy_compile_target}" MODULE "${dst_c}")
  get_property(cython_link_libs SOURCE "${SOURCE_FILE}"
    PROPERTY CYTHON_LINK_LIBRARIES)
  if(cython_link_libs)
    target_link_libraries("${_cy_compile_target}" ${cython_link_libs})
  endif()
  set_target_properties("${_cy_compile_target}"
    PROPERTIES POSITION_INDEPENDENT_CODE ON
    LIBRARY_OUTPUT_NAME "${src_base}"
    PREFIX ""
    SUFFIX "${PYTHON_EXTENSION_SUFFIX}"
    LIBRARY_OUTPUT_DIRECTORY "${dst_path}")
  if(ENABLE_CYTHON_COVERAGE)
    set_property(TARGET "${_cy_compile_target}"
      APPEND PROPERTY COMPILE_DEFINITIONS CYTHON_TRACE_NOGIL=1 CYTHON_TRACE=1)
  endif()
  target_include_directories("${_cy_compile_target}" PRIVATE ${Python_INCLUDE_DIRS})
  add_dependencies("${_cy_compile_target}" "${_cy_compile_target}-c" ${pyx_dep})
  add_dependencies(all-python-target "${_cy_compile_target}")

  set(${OUT_PYX} "${dst}" PARENT_SCOPE)
  if(EXISTS "${src_pxd}")
    set(${OUT_PXD} "${dst_pxd}" PARENT_SCOPE)
  else()
    set(${OUT_PXD} "" PARENT_SCOPE)
  endif()
  set(${OUT_C} "${dst_c}" PARENT_SCOPE)
  set(${OUT_SO} "${dst_so}" PARENT_SCOPE)
  set(${OUT_TGT} "${_cy_compile_target}" PARENT_SCOPE)
endfunction()

function(_cython_copy_pxd SOURCE_FILE OUT_PXD)
  # Filename
  get_filename_component(src_base "${SOURCE_FILE}" NAME_WE)

  cmake_utils_abs_path(src_pxd "${SOURCE_FILE}")
  get_filename_component(src_path "${src_pxd}" PATH)
  cmake_utils_is_subpath(issub "${CMAKE_BINARY_DIR}" "${src_pxd}")

  if(issub)
    set(${OUT_PXD} "${src_pxd}" PARENT_SCOPE)
    return()
  endif()

  cmake_utils_src_to_bin(dst_pxd "${src_pxd}")
  cmake_utils_is_subpath(issub "${CMAKE_BINARY_DIR}" "${dst_pxd}")
  if(NOT issub)
    message(FATAL_ERROR "Cannot determine binary directory for ${src_pxd}")
  endif()
  get_filename_component(dst_path "${dst_pxd}" PATH)
  file(MAKE_DIRECTORY "${dst_path}")
  file(RELATIVE_PATH rel_path "${_BASE_DIR}" "${src_path}/${src_base}")
  string(REGEX REPLACE "[^-._a-z0-9A-Z]" "_" target_name "${rel_path}")
  cmake_utils_get_unique_target(cython-${target_name} _cy_compile_target)

  # Copying the sources files over
  add_custom_command(
    OUTPUT "${dst_pxd}"
    COMMAND ${CMAKE_COMMAND} -E copy "${src_pxd}" "${dst_pxd}"
    DEPENDS "${src_pxd}")
  add_custom_target("${_cy_compile_target}-pxd"
    DEPENDS "${dst_pxd}")
  add_dependencies(cython-all-pxd-target "${_cy_compile_target}-pxd")

  set(${OUT_PXD} "${dst_pxd}" PARENT_SCOPE)
endfunction()

macro(__python_compile)
  get_filename_component(_ext "${_pyfile}" EXT)
  if("${_ext}" STREQUAL ".py")
    _python_compile("${_pyfile}" out_py out_pyc out_tgt)
  elseif("${_ext}" STREQUAL ".pyx")
    _cython_compile("${_pyfile}" out_pyx out_pxd out_c out_so out_tgt)
  elseif("${_ext}" STREQUAL ".pxd")
    _cython_copy_pxd("${_pyfile}" out_pxd)
  else()
    message(FATAL_ERROR "Unknown file type ${_pyfile}")
  endif()
endmacro()

function(python_compile _BASE_DIR)
  cmake_array_foreach(_pyfile __python_compile)
endfunction()

macro(__python_install)
  get_filename_component(_ext "${_pyfile}" EXT)
  if("${_ext}" STREQUAL ".py")
    _python_compile("${_pyfile}" out_py out_pyc out_tgt)
    install(FILES "${out_py}" DESTINATION "${DEST_DIR}")
    set(PYC_DEST_DIR "${DEST_DIR}/__pycache__")
    install(FILES "${out_pyc}" DESTINATION "${PYC_DEST_DIR}")
  elseif("${_ext}" STREQUAL ".pyx")
    _cython_compile("${_pyfile}" out_pyx out_pxd out_c out_so out_tgt)
    if(NOT "${out_pxd}" STREQUAL "")
      install(FILES "${out_pxd}" DESTINATION "${DEST_DIR}")
    endif()
    install(TARGETS "${out_tgt}" DESTINATION "${DEST_DIR}")
  elseif("${_ext}" STREQUAL ".pxd")
    _cython_copy_pxd("${_pyfile}" out_pxd)
    install(FILES "${out_pxd}" DESTINATION "${DEST_DIR}")
  else()
    message(FATAL_ERROR "Unknown file type ${_pyfile}")
  endif()
endmacro()

function(python_install _BASE_DIR DEST_DIR)
  cmake_array_foreach(_pyfile __python_install 1)
endfunction()

function(python_install_as_module _BASE_DIR)
  set(DEST_DIR "${_PYTHON_SITE_INSTALL_DIR}")
  cmake_array_foreach(_pyfile __python_install 1)
endfunction()

function(python_install_module _BASE_DIR MODULE_NAME)
  set(DEST_DIR "${_PYTHON_SITE_INSTALL_DIR}/${MODULE_NAME}")
  cmake_array_foreach(_pyfile __python_install 2)
endfunction()

macro(__python_test)
  get_filename_component(_ext "${_pyfile}" EXT)
  if("${_ext}" STREQUAL ".py")
    _python_compile("${_pyfile}" out_py out_pyc out_tgt)
    get_filename_component(src_base "${_pyfile}" NAME_WE)
    if(ENABLE_CYTHON_COVERAGE)
      set(coverage_args -m coverage run
        "--rcfile=${_py_cmake_module_dir}/coveragerc" -p)
    else()
      set(coverage_args)
    endif()
    if("${src_base}" MATCHES "^test_")
      add_test(NAME test/python/${src_base}
        COMMAND env "PYTHONPATH=${PYTHONPATH}"
        ${Python_EXECUTABLE} ${coverage_args} -m pytest "${out_py}"
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")
    endif()
  elseif("${_ext}" STREQUAL ".pyx")
    # No automatic test for pyx files since pytest doesn't support it
    _cython_compile("${_pyfile}" out_pyx out_pxd out_c out_so out_tgt)
  elseif("${_ext}" STREQUAL ".pxd")
    _cython_copy_pxd("${_pyfile}" out_pxd)
  else()
    message(FATAL_ERROR "Unknown file type ${_pyfile}")
  endif()
endmacro()

function(python_test PYTHONPATH)
  # Setting this makes sure that the library pxd's are in a sub directory
  # and therefore have their filename stored as a relative path rather than
  # just a filename (as is the case for system headers).
  # This makes sure that the file path is correct in the coverage report.
  set(_BASE_DIR "${CMAKE_BINARY_DIR}")
  set(PYTHONPATH "${PYTHONPATH}:${CMAKE_CURRENT_BINARY_DIR}")
  cmake_array_foreach(_pyfile __python_test 1)
endfunction()
