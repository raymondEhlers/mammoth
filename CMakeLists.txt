cmake_minimum_required(VERSION 3.14..3.18)
project(mammoth
  LANGUAGES
		CXX
)

# Only do these if this is the main project, and not if it is included through
# add_subdirectory
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  # Optionally set things like CMAKE_CXX_STANDARD,
  # CMAKE_POSITION_INDEPENDENT_CODE here

  # Let's ensure -std=c++xx instead of -std=g++xx
  set(CMAKE_CXX_EXTENSIONS OFF)

  # Let's nicely support folders in IDE's
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)

  # Use RPath on Mac OS.
  set(CMAKE_MACOSX_RPATH 1)
endif()

# Path to additonal modules
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

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

# Setup mammothExt, which includes jet finding functionality.
# c++17 is required for std::optional
add_library(mammothExt mammoth/src/jetFinding.cxx)
target_compile_features(mammothExt PUBLIC cxx_std_17)
target_link_libraries(mammothExt PUBLIC FastJet::FastJet FastJet::Contrib)
target_include_directories(mammothExt PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include" FastJet::FastJet)

# Create the pybind11 bindings
pybind11_add_module(_ext mammoth/src/_ext.cxx)
target_link_libraries(_ext PRIVATE mammothExt)
target_include_directories(_ext PUBLIC mammothExt)