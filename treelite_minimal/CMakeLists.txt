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

set(SOURCES src/builder.cc src/model.cc src/optable.cc src/serializer.cc
            include/treelite/base.h include/treelite/frontend.h include/treelite/tree.h)

add_library(treelite STATIC ${SOURCES})

target_include_directories(treelite PUBLIC include PRIVATE 3rdparty/dmlc-core/include)
target_link_libraries(treelite PUBLIC dmlc)

add_executable(tester src/tester.cc)
target_include_directories(tester PUBLIC include PRIVATE 3rdparty/dmlc-core/include)
target_link_libraries(tester PRIVATE treelite)

set_target_properties(treelite tester
  PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED ON)

set_property(TARGET tester APPEND_STRING PROPERTY COMPILE_FLAGS -fsanitize=address)
set_property(TARGET tester APPEND_STRING PROPERTY LINK_FLAGS -fsanitize=address)

set_default_configuration_release()
msvc_use_static_runtime()
