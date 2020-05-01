# Automatically set source group based on folder
function(auto_source_group SOURCES)

  foreach(FILE ${SOURCES})
    get_filename_component(PARENT_DIR "${FILE}" PATH)

    # skip src or include and changes /'s to \\'s
    string(REPLACE "${CMAKE_CURRENT_LIST_DIR}" "" GROUP "${PARENT_DIR}")
    string(REPLACE "/" "\\\\" GROUP "${GROUP}")
    string(REGEX REPLACE "^\\\\" "" GROUP "${GROUP}")

    source_group("${GROUP}" FILES "${FILE}")
  endforeach()
endfunction(auto_source_group)

# Force static runtime for MSVC
function(msvc_use_static_runtime)
  if(MSVC)
    set(variables
        CMAKE_CXX_FLAGS_DEBUG
        CMAKE_CXX_FLAGS_MINSIZEREL
        CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_RELWITHDEBINFO
    )
    foreach(variable ${variables})
      if(${variable} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${variable} "${${variable}}")
        set(${variable} "${${variable}}"  PARENT_SCOPE)
      endif()
    endforeach()
  endif()
endfunction(msvc_use_static_runtime)

# Set a default build type to release if none was specified
function(set_default_configuration_release)
  if(CMAKE_CONFIGURATION_TYPES STREQUAL "Debug;Release;MinSizeRel;RelWithDebInfo") # multiconfig generator?
    set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "" FORCE)
  elseif(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE )
  endif()
endfunction(set_default_configuration_release)
