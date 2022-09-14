"""
Adapted from https://github.com/pybind/cmake_example
"""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from setuptools import Distribution, Extension
from setuptools.command.build_ext import build_ext

# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}

# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):  # type: ignore
    def __init__(self, name: str, sourcedir: str = ""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):  # type: ignore
    def build_extension(self, ext: CMakeExtension) -> None:  # noqa: C901
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        # CMake lets you override the generator - we need to check this.
        # Can be set with Conda-Build, for example.
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        # EXAMPLE_VERSION_INFO shows you how to pass a value into the C++ code
        # from Python.
        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}".format(extdir),
            "-DPYTHON_EXECUTABLE={}".format(sys.executable),
            # We don't need this, and it causes a CMake warning, so comment out for now.
            # But we'll keep it around in case we need it later, since it seems to be
            # potentially quite useful.
            #"-DEXAMPLE_VERSION_INFO={}".format(self.distribution.get_version()),
            "-DCMAKE_BUILD_TYPE={}".format(cfg),  # not used on MSVC, but no harm
        ]
        build_args = []

        # Adding CMake arguments set as environment variable
        # (needed e.g. to build for ARM OSx on conda-forge)
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        if self.compiler.compiler_type != "msvc":
            # Using Ninja-build since it a) is available as a wheel and b)
            # multithreads automatically. MSVC would require all variables be
            # exported for Ninja to pick it up, which is a little tricky to do.
            # Users can override the generator with CMAKE_GENERATOR in CMake
            # 3.15+.
            if not cmake_generator or cmake_generator == "Ninja":
                try:
                    import ninja  # noqa: F401

                    ninja_executable_path = os.path.join(ninja.BIN_DIR, "ninja")
                    cmake_args += [
                        "-GNinja",
                        f"-DCMAKE_MAKE_PROGRAM:FILEPATH={ninja_executable_path}",
                    ]
                except ImportError:
                    pass

        else:

            # Single config generators are handled "normally"
            single_config = any(x in cmake_generator for x in {"NMake", "Ninja"})

            # CMake allows an arch-in-generator style for backward compatibility
            contains_arch = any(x in cmake_generator for x in {"ARM", "Win64"})

            # Specify the arch if using MSVC generator, but only if it doesn't
            # contain a backward-compatibility arch spec already in the
            # generator name.
            if not single_config and not contains_arch:
                cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]

            # Multi-config generators have a different way to specify configs
            if not single_config:
                cmake_args += [
                    "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir)
                ]
                build_args += ["--config", cfg]

        if sys.platform.startswith("darwin"):
            # Cross-compile support for macOS - respect ARCHFLAGS if set
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += ["-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += ["-j{}".format(self.parallel)]

        build_temp = os.path.join(self.build_temp, ext.name)
        print(f"{build_temp=}")
        if not os.path.exists(build_temp):
            os.makedirs(build_temp)

        print(f"{ext.sourcedir=}, {cmake_args=}, {build_temp=}")
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=build_temp
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=build_temp
        )


def build(setup_kwargs: Mapping[str, Any]) -> None:
    # Setup the distribution
    distribution = Distribution({"name": "mammoth", "ext_modules": [CMakeExtension("mammoth._ext")]})
    distribution.package_dir = {"": "mammoth"}

    # More or less directly from https://github.com/python-poetry/poetry/issues/2740
    cmd = CMakeBuild(distribution, debug=True)
    cmd.ensure_finalized()
    cmd.run()

    # Copy built extensions back to the project
    # We want to retrieve everything from the cmd outputs (which contains the pybind11 bindings),
    output_directories = set(Path(p).parent for p in cmd.get_outputs())
    # However, we also need the additional libraries that we've build.
    # We can add those by looking for libraries in the same directory as the bindings.
    libraries_to_move = [Path(p) for p in cmd.get_outputs()] + \
                        [p for output_dir in output_directories for p in Path(output_dir).glob("lib*")]
    for output in libraries_to_move:
        relative_extension = os.path.relpath(output, cmd.build_lib)
        # First, remove the existing to avoid any conflicts with compilers...
        # This is redundant, but for some reason, it seems to make segfaults less common on macOS.
        # Best guess is that by removing and copying, it perhaps invalidates some cache mechanism that isn't
        # invalidated when just copying. But this is pure speculation. Since it works, it's not worth worrying
        # about further
        Path(relative_extension).unlink(missing_ok=True)

        # And now actually copy
        print(f"Copying {output} to {relative_extension}")
        shutil.copyfile(output, relative_extension)
        mode = os.stat(relative_extension).st_mode
        mode |= (mode & 0o444) >> 2
        os.chmod(relative_extension, mode)


if __name__ == "__main__":
    build({})
