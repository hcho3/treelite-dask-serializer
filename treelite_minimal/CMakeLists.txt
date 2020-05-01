cmake_minimum_required (VERSION 3.10)
project(treelite)
include(cmake/Utils.cmake)

# Use RPATH on Mac OS X as flexible mechanism for locating dependencies
# See https://blog.kitware.com/upcoming-in-cmake-2-8-12-osx-rpath-support/
set(CMAKE_MACOSX_RPATH TRUE)

# check MSVC version
if(MSVC)
  if(MSVC_VERSION LESS 1900)
    message(FATAL_ERROR "Need Visual Studio 2015 or newer to compile treelite")
  endif()
endif()

add_subdirectory(3rdparty/dmlc-core)

file(GLOB_RECURSE SOURCES
  src/*.cc
  src/*.h
  include/*.h
)

add_library(treelite STATIC ${SOURCES})

set_target_properties(treelite
  PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED ON)

target_include_directories(treelite PUBLIC include PRIVATE 3rdparty/dmlc-core/include)
target_link_libraries(treelite PUBLIC dmlc)

set_default_configuration_release()
msvc_use_static_runtime()
