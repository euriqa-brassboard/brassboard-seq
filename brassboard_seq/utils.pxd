# cython: language_level=3

from cpython cimport PyObject

cdef extern from "src/utils.h":
    T assume[T](T) noexcept nogil
    void assume_not_none(object) noexcept nogil
    void _assume_not_none "assume_not_none"(void*) noexcept nogil
