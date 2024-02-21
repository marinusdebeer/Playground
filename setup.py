from setuptools import setup, Extension

module = Extension('multiply', sources=['multiply.c'])

setup(name='MultiplyModule',
      version='1.0',
      description='Python wrapper for multiplying two numbers in C',
      ext_modules=[module])
