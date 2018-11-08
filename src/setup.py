from distutils.core import setup, Extension

SRC = [
    'pyhack-module.c',
    'wrapper-cblas.c',
    'symbols.c',
    'wrapper.c'
]

setup(name='pyhack', version='1.0', ext_modules=[Extension('pyhack', SRC)])
