cmake_minimum_required(VERSION 3.19...3.30)

# Help out running the case that we're not using scikit-build-core (e.g. we're trying to use vscode)
# If we're using scikit-build, this will do nothing.
if(NOT DEFINED SKBUILD)
  set(SKBUILD_PROJECT_NAME mammoth_cpp)
  set(SKBUILD_PROJECT_VERSION 0.0.0)
endif()

project("${SKBUILD_PROJECT_NAME}" LANGUAGES CXX VERSION "${SKBUILD_PROJECT_VERSION}")

##########################
# Step 1: Project settings
##########################
# Cross check from awkward-cpp
if(CMAKE_CXX_COMPILER_ID MATCHES AppleClang AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL
                                                15)
  if(CMAKE_VERSION VERSION_LESS 3.29)
    message(WARNING "CMake should be 3.29+ to support AppleClang 15+. Trying anyway.")
  endif()
  set(CMAKE_LINKER_TYPE APPLE_CLASSIC)
endif()

# Ensure c++17
# Selected 17 because we use std::optional
# We specify this down below with an apparently preferred way (see: https://cliutils.gitlab.io/modern-cmake/chapters/features/cpp11.html)
# However, as of September 2022, this different way doesn't always seem to work as expected (maybe), so I keep these
# options enabled here, since they do always seem to work. If I made this into a proper library, this would be an issue
# and should be resolved in that case. But since this is just for my own use, there's no need to go further
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

# Only do these if this is the main project, and not if it is included through
# add_subdirectory
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  # Optionally set things like CMAKE_CXX_STANDARD,
  # CMAKE_POSITION_INDEPENDENT_CODE here

  # Let's nicely support folders in IDE's
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)

endif()

# Some rpath defaults taken from awkward-cpp.
# Further helpful docs are here: https://jeremimucha.com/2021/04/cmake-fundamentals-part8/
# including both details on how to understand rpath, as well as useful conventions.
if(APPLE)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
  set(CMAKE_INSTALL_RPATH "@loader_path")
else()
  set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)
  # NOTE: The second entry is needed due to the fastjet install procedure where we copy to mammoth_cpp below.
  #       It's a bit awkward that this is defined here before those steps, but we need to set the rpath
  #       before doing anything else, so we just have to live with it.
  set(CMAKE_INSTALL_RPATH $ORIGIN $ORIGIN/fastjet/lib)
endif()

# Path to additional cmake modules
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# Let the user know the build type
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# NOTE: If we need to enable address sanitization
#add_compile_options(-fno-omit-frame-pointer -fsanitize=address)
#add_link_options(-fno-omit-frame-pointer -fsanitize=address)

##########################
# Step 2: Build libmammoth
##########################
# This contains c++ functionality, not including the python bindings

# Require fastjet
find_package(FastJet)
if(NOT FastJet_FOUND)
    # Install using our install script if it's not already installed.
    # NOTE: As of Nov 2022, this doesn't seem to pick up the right compiler on macOS.
    #       But it's easy enough to build, so good enough for now
    execute_process(
        COMMAND ${PROJECT_SOURCE_DIR}/external/install_fastjet.sh
        RESULT_VARIABLE cmd_result
        OUTPUT_VARIABLE cmd_output
        ERROR_VARIABLE cmd_error
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE
    )
    if(NOT cmd_result EQUAL 0)
        message(FATAL_ERROR "Installing fastjet failed with return code: ${cmd_result}\nOutput: ${cmd_output}\nError: ${cmd_error}")
    endif()

    # Now it must be there - find it.
    find_package(FastJet REQUIRED)
endif()

## boost (for EECs)
## NOTE: We skip this for now since the code seems rather tricky to use.
##       Better to just keep it simple for now. But we also keep this here
##       for possible future use.
#find_package(Boost)
#if(NOT Boost_FOUND)
#  include("${CMAKE_MODULE_PATH}/DownloadBoostHistogram.cmake")
#  download_boost_histogram(BOOST_TAG 1.81.0)
#endif()
## They're now found - no need to do find_package again since histogram is header only
#message(STATUS "POST: BH_INCLUDE_DIRS: ${BH_INCLUDE_DIRS}")

# Setup libmammoth, which includes jet finding functionality.
file(GLOB MAMMOTH_SOURCES "src/libmammoth/*.cxx")
add_library(mammoth SHARED ${MAMMOTH_SOURCES})
# Set c++17 explicitly, but don't use compiler extensions
target_compile_features(mammoth PUBLIC cxx_std_17)
set_target_properties(mammoth PROPERTIES CXX_EXTENSIONS OFF)
# Add version info
target_compile_definitions(mammoth
                           PRIVATE VERSION_INFO=${PROJECT_VERSION})
# Add warnings
target_compile_options(mammoth PRIVATE -Wall -Wextra -pedantic -Werror)
# Setup linking and includes
target_link_libraries(mammoth PUBLIC FastJet::FastJet FastJet::Contrib)
target_include_directories(mammoth PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include" FastJet::FastJet)

# And install
install(TARGETS mammoth LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})

##############################
# Step 2: Onto the Python part
##############################

# First, grab python, as recommended here: https://scikit-build-core.readthedocs.io/en/latest/getting_started.html
# NOTE: On hiccup, this sometimes caused an issue (maybe something about pyenv? Unclear). If that happens, then just comment out this line.
#       It's not ideal from a correct configuration perspective, but the build works and all the tests pass, so it all seems to be fine...
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
# pybind11 from PyPI, which we require in the build dependencies.
# NOTE: pybind11 seems to be the package that defines PYTHON_SITE_PACKAGES
find_package(pybind11 CONFIG REQUIRED)

# If we need to mammoth-cpp site-packages directory, it's available here.
# We don't for now, but we leave it here for our notes, because this wasn't so easy to find the first time.
#set(MAMMOTH_CPP_LIB_PATH "${PYTHON_SITE_PACKAGES}/${SKBUILD_PROJECT_NAME}")
#message(STATUS "Installing to: ${MAMMOTH_CPP_LIB_PATH}")

# Create the pybind11 bindings
file(GLOB MAMMOTH_EXT_SOURCES "src/python/*.cxx")
pybind11_add_module(_ext ${MAMMOTH_EXT_SOURCES})
# Set c++17 explicitly, but don't use compiler extensions
target_compile_features(_ext PUBLIC cxx_std_17)
set_target_properties(_ext PROPERTIES CXX_EXTENSIONS OFF)
# Add version info
target_compile_definitions(_ext
                           PRIVATE VERSION_INFO=${PROJECT_VERSION})
# Add warnings
target_compile_options(_ext PRIVATE -Wall -Wextra -pedantic -Werror)
# Setup linking and includes
target_link_libraries(_ext PUBLIC mammoth)
target_include_directories(_ext PUBLIC mammoth)

add_custom_command(TARGET _ext POST_BUILD
    # Provide explicit notification if the command fails
    COMMAND_ERROR_IS_FATAL ANY
    # Apparently this is best practices to avoid any unexpected issues.
    VERBATIM
    # Copy the fastjet libraries to the mammoth install dir (ie. mammoth-cpp/fastjet)
    # By doing so, we can add it via the sdist inclusion, and then pip will manage it in site-packages.
    # After all of these steps, fj libs should always be accessible for our libraries.
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${FastJet_LIB_DIRS}/../.." "${PROJECT_SOURCE_DIR}/src/${SKBUILD_PROJECT_NAME}"
    # Address the fact that fastjet-config unfortunately provides the wrong path (providing where it was actually built
    # rather than the new location), so we update the script.
    # NOTE: We don't yet provide access to this script, and it apparently isn't marked as executable.
    #       If we want to provide access, we should provide an entry point that wraps this script.
    #       As of November 2022, this isn't a high priority, so we leave it as is
    COMMAND bash -c "${PROJECT_SOURCE_DIR}/external/repair_fastjet_config_path.sh ${PROJECT_SOURCE_DIR}/src/${SKBUILD_PROJECT_NAME}/fastjet/bin/fastjet-config"
)

# And finally install the bindings
install(TARGETS _ext LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})
