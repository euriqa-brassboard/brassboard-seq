# cython: language_level=3

cimport cython
from cpython cimport PyErr_Format

@cython.final
cdef class RuntimeValue:
    def __init__(self):
        # All instances should be constructed within cython code via
        # `RuntimeValue.__new__` or its wrapper.
        PyErr_Format(TypeError, "RuntimeValue cannot be created directly")
