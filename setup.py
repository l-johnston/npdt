"""Numpy's new DType"""
import os
import sys
import pathlib
import sysconfig
from setuptools import setup, Extension


def find_numpy_include():
    """find numpy's include directory

    Can't use np.get_include() when inside virtual environment
    """
    path = pathlib.Path(sys.prefix).joinpath("lib")
    path = [*path.glob("python*")][0]
    path = path.joinpath("site-packages/numpy/core/include")
    if path.exists():
        return path
    raise RuntimeError("numpy include path not found")


# pylint: disable=invalid-name
version = "0.0.1"
extra_compile_args = ["-O3", "-w"]
extensions = [
    Extension(
        name="npdt.customfloat",
        sources=["src/customfloat.c"],
        depends=["src/customfloat.c"],
        include_dirs=[
            "src",
            sysconfig.get_paths()["include"],
            find_numpy_include(),
        ],
        extra_compile_args=extra_compile_args,
    ),
]
setup_metadata = {
    "name": "npdt",
    "packages": ["npdt"],
    "package_dir": {"": "src"},
    "ext_modules": extensions,
    "install_requires": ["numpy"],
    "version": version,
}
if __name__ == "__main__":
    os.system("export NUMPY_EXPERIMENTAL_DTYPE_API=1")
    setup(**setup_metadata)
