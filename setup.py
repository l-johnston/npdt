"""Numpy's new DType"""
import os
import sys
import pathlib
import sysconfig
from setuptools import setup, Extension

import numpy as np


# pylint: disable=invalid-name
version = "0.0.1"
extra_compile_args = ["-O3", "-w"]
extensions = [
    Extension(
        name="npdt.customfloat",
        sources=["src/customfloat.c"],
        include_dirs=[np.get_include(),],
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
