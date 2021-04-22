import os
import subprocess

import numpy as np
from Cython.Build import build_ext
from setuptools import find_packages, setup, Extension


# Include libraries from the OS X Command Line Tools. On OS X Big Sur, these libraries
# are not automatically included anymore.
osx_library_path = "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib"
if os.path.exists(osx_library_path):
    if "LIBRARY_PATH" in os.environ and os.environ["LIBRARY_PATH"]:
        os.environ["LIBRARY_PATH"] += ":" + osx_library_path
    else:
        os.environ["LIBRARY_PATH"] = osx_library_path

# If `xcrun` is available, make sure the includes are added to CPATH.
if subprocess.call("which xcrun", shell=True) == 0:
    path = (
        subprocess.check_output("xcrun --show-sdk-path", shell=True)
        .strip()
        .decode("ascii")
    )
    path += "/usr/include"

    # Add to CPATH.
    if "CPATH" not in os.environ:
        os.environ["CPATH"] = ""
    os.environ["CPATH"] += path

# Default to use gcc as the compiler if `$CC` is not set.
if "CC" not in os.environ or not os.environ["CC"]:
    os.environ["CC"] = "gcc"

# Check whether `gfortran` is available.
if subprocess.call("which gfortran", shell=True) != 0:
    gfortran_available = False
else:
    gfortran_available = True

# Ensure that `$CC` is not symlinked to `clang`, because the default shipped
# one often does not support OpenMP, but `gcc` does.
out = subprocess.check_output("$CC  --version", shell=True)
if "clang" in out.decode("ascii"):
    # It is. Now try to find a `gcc` to replace it with.
    found = False
    for i in range(100, 3, -1):
        gcci = "gcc-{}".format(i)
        if subprocess.call(["which", gcci]) == 0:
            # Set both `$CC` and `$CXX` in this case, just to be sure.
            os.environ["CC"] = gcci
            os.environ["CXX"] = "g++-{}".format(i)
            found = True
            break

    # Ensure that one was found.
    if not found:
        raise RuntimeError(
            "Your gcc runs clang, and no version of gcc could be found. "
            "Please install gcc. "
            'On OS X, this can be done with "brew install gcc".'
        )

# Compile TVPACK if `gfortran` is available.
if gfortran_available:
    if (
        subprocess.call(
            "gfortran -fPIC -O2 -c lab/bvn_cdf/tvpack.f -o lab/bvn_cdf/tvpack.o",
            shell=True,
        )
        != 0
    ):
        raise RuntimeError("Compilation of TVPACK failed.")

requirements = ["numpy>=1.16", "scipy>=1.3", "fdm", "plum-dispatch>=1"]


# Determine which external modules to compile.
ext_modules = []

if gfortran_available:
    ext_modules.append(
        Extension(
            "lab.bvn_cdf",
            sources=["lab/bvn_cdf/bvn_cdf.pyx"],
            include_dirs=[np.get_include()],
            extra_compile_args=["-fPIC", "-O2", "-fopenmp"],
            extra_objects=["lab/bvn_cdf/tvpack.o"],
            extra_link_args=["-lgfortran", "-fopenmp"],
        )
    )

setup(
    packages=find_packages(exclude=["docs"]),
    python_requires=">=3.6",
    install_requires=requirements,
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules,
    include_package_data=True,
)
