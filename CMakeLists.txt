cmake_minimum_required(VERSION 3.15..3.18)
project(mammoth
    LANGUAGES
        CXX
    )

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

# Path to additional modules
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# First, setup python, using the new FindPython mode
# Added according to this ref: https://pybind11.readthedocs.io/en/stable/cmake/index.html#new-findpython-mode
# As a plus, it also fixes an issue in vscode running CMake
find_package(Python COMPONENTS Interpreter Development)

# pybind11 from PyPI
#execute_process(
#  COMMAND
#    "${PYTHON_EXECUTABLE}" -c
#    "import pybind11; print(pybind11.get_cmake_dir())"
#  OUTPUT_VARIABLE _tmp_dir
#  OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
#list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")
## Require pybind11
#find_package(pybind11 CONFIG REQUIRED)

# Alternatively, pybind11 from a submodule
add_subdirectory(external/pybind11)

# Require fastjet
find_package(FastJet REQUIRED)

# If we need to enable address sanitization
#add_compile_options(-fno-omit-frame-pointer -fsanitize=address)
#add_link_options(-fno-omit-frame-pointer -fsanitize=address)

# Setup mammothExt, which includes jet finding functionality.
# c++17 is required for std::optional
add_library(mammothext SHARED mammoth/src/jetFinding.cxx mammoth/src/aliceFastSim.cxx)
target_compile_features(mammothext PUBLIC cxx_std_17)
set_target_properties(mammothext PROPERTIES CXX_EXTENSIONS OFF)
# Add warnings
target_compile_options(mammothext PRIVATE -Wall -Wextra -pedantic -Werror)
target_link_libraries(mammothext PUBLIC FastJet::FastJet FastJet::Contrib)
target_include_directories(mammothext PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include" FastJet::FastJet)

# Create the pybind11 bindings
pybind11_add_module(_ext mammoth/src/_ext.cxx)
target_compile_features(_ext PUBLIC cxx_std_17)
set_target_properties(_ext PROPERTIES CXX_EXTENSIONS OFF)
# Add warnings
target_compile_options(_ext PRIVATE -Wall -Wextra -pedantic -Werror)
target_link_libraries(_ext PUBLIC mammothext)
target_include_directories(_ext PUBLIC mammothext)
