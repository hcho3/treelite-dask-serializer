import os
from distutils.sysconfig import get_python_lib
from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Distutils.build_ext import new_build_ext as build_ext

extensions = [
    Extension('*',
              sources=['treelite_dask_serializer/*.pyx'],
              include_dirs=['treelite_minimal/include',
                            'treelite_minimal/3rdparty/dmlc-core/include'],
              library_dirs=[get_python_lib(), 'treelite_minimal/build/'],
              runtime_library_dirs=[os.path.join(os.sys.prefix, 'lib')],
              libraries=['treelite'],
              language='c++',
              extra_compile_args=['--std=c++11'])
]

setup(
    name='treelite_dask_serializer',
    version='0.0.1',
    setup_requires=['cython'],
    ext_modules=extensions,
    packages=find_packages(),
    install_requires=['cython'],
    cmdclass={'build_ext': build_ext},
    zip_safe=False
)
