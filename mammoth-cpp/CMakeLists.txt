cmake_minimum_required(VERSION 3.15..3.24)
project("${SKBUILD_PROJECT_NAME}" LANGUAGES CXX VERSION "${SKBUILD_PROJECT_VERSION}")

# Ensure c++17
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

  # Use RPath on Mac OS.
  # Commented out as of September 2022. It maybe causes issues with newer macOS versions (I seem memory corruption with an unclear cause
  # since it doesn't appear on linux). If it comes back, or RPATH appears to be required, check out using an address sanitizer.
  # See: https://stackoverflow.com/questions/44320465/whats-the-proper-way-to-enable-addresssanitizer-in-cmake-that-works-in-xcode
  #set(CMAKE_MACOSX_RPATH 1)
endif()

# Some rpath defaults taken from awkward-cpp.
if(APPLE)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
  set(CMAKE_INSTALL_RPATH "@loader_path")
else()
  set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)
endif()

# Path to additional modules
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Let the user know
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# NOTE: If we need to enable address sanitization
#add_compile_options(-fno-omit-frame-pointer -fsanitize=address)
#add_link_options(-fno-omit-frame-pointer -fsanitize=address)

##########################
# Step 1: Build libmammoth
##########################

# Require fastjet
find_package(FastJet)
if(NOT FastJet_FOUND)
    # Install using our install script if it's not already installed.
    execute_process(COMMAND external/install_fastjet.sh)
    # Now it must be there
    find_package(FastJet REQUIRED)
endif()

# Setup libmammoth, which includes jet finding functionality.
# c++17 is required for std::optional
file(GLOB MAMMOTH_SOURCES "src/libmammoth/*.cxx")
add_library(mammoth SHARED ${MAMMOTH_SOURCES})
target_compile_features(mammoth PUBLIC cxx_std_17)
# Add version info
target_compile_definitions(mammoth
                           PRIVATE VERSION_INFO=${PROJECT_VERSION})
set_target_properties(mammoth PROPERTIES CXX_EXTENSIONS OFF)
# Add warnings
target_compile_options(mammoth PRIVATE -Wall -Wextra -pedantic -Werror)
# Setup linking and includes
target_link_libraries(mammoth PUBLIC FastJet::FastJet FastJet::Contrib)
target_include_directories(mammoth PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include" FastJet::FastJet)

# And install
install(TARGETS mammoth DESTINATION ${SKBUILD_PROJECT_NAME})

##############################
# Step 2: Onto the Python part
##############################

# pybind11 from PyPI via scikit-build-core
# NOTE: This seems to define PYTHON_SITE_PACKAGES
find_package(pybind11 CONFIG REQUIRED)

set(MAMMOTH_CPP_LIB_PATH "${PYTHON_SITE_PACKAGES}/${SKBUILD_PROJECT_NAME}")
message(STATUS "Installing to: ${MAMMOTH_CPP_LIB_PATH}")

# Create the pybind11 bindings
file(GLOB MAMMOTH_EXT_SOURCES "src/python/*.cxx")
pybind11_add_module(_ext ${MAMMOTH_EXT_SOURCES})
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

# Copy the fastjet libraries to the mammoth install dir (ie. mammoth-cpp/fastjet)
add_custom_command(TARGET _ext POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${FastJet_LIB_DIRS}/../.." "${CMAKE_SOURCE_DIR}/src/${SKBUILD_PROJECT_NAME}"
)
# NOTE: This doesn't work because the files aren't copied until later in the build. It's a nice idea, but
#       it needs some more thought on how to do it.
## And update the fastjet-config file so that it points to valid paths.
#file(READ ${MAMMOTH_CPP_MAMMOTH_CPP_LIB_PATH}/fastjet/bin/fastjet-config filedata)
#string(REGEX REPLACE "^prefix\=.+$" "prefix=${MAMMOTH_CPP_LIB_PATH}/fastjet\n" filedata "${filedata}")
## As far as I can tell, it always appears to be the same as prefix, so we assume it is here.
#string(REGEX REPLACE "^installationdir\=.+$" "installationdir=${MAMMOTH_CPP_LIB_PATH}/fastjet\n" filedata "${filedata}")
#file(WRITE ${MAMMOTH_CPP_LIB_PATH}/fastjet/bin/fastjet-config "${filedata}")

# And finally install
install(TARGETS _ext DESTINATION ${SKBUILD_PROJECT_NAME})

