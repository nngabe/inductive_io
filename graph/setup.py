# Author: Nicholas Gabriel <ngabriel (at) gwu.edu>, Mar. 2021
#
# License: GNU General Public License v3.0
from setuptools import Extension,setup
from Cython.Build import cythonize
import numpy
import os

mod0 = Extension('blas', ['blas.pyx'],
        include_dirs=[],
        libraries=[],
        library_dirs=[],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
        extra_compile_args=['-O3'])#,"-ffast-math","-fno-trapping-math","-funroll-loops"])

mod1 = Extension('optimize', ['optimize.pyx'],
        include_dirs=[],
        libraries=[],
        library_dirs=[],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
        extra_compile_args=['-O3','-ffast-math','-fno-trapping-math','-funroll-loops'])

modules = [mod0,mod1]

setup(
    ext_modules=cythonize(modules,language_level='3'),
    include_dirs=[numpy.get_include()]

)
