from distutils.core import setup, Extension

SRC = [
    'pyhack-module.c',
    'wrapper-cblas.c',
    'symbols.c',
    'wrapper.c'
]

INCLUDE_PATHS = [
    '/usr/local/cuda/include/'
]

LIBS = [
    'cublas',
    'cuda',
    'cudart'
]

LIBS_PATH = [
    '/usr/local/cuda/lib64/'
]

setup(name='pyhack', version='1.0', ext_modules=[
    Extension('pyhack', SRC, include_dirs=INCLUDE_PATHS,
              libraries=LIBS, library_dirs=LIBS_PATH)
])
