# cython: language_level=3

from cpython cimport PyObject

cdef extern from *:
    """
#include "Python.h"

#ifdef __has_builtin
#  define bb_has_builtin(x) __has_builtin(x)
#else
#  define bb_has_builtin(x) 0
#endif

#if bb_has_builtin(__builtin_assume)
template<typename T>
static inline T assume(T v)
{
    __builtin_assume(bool(v));
    return v;
}
#elif defined(__GNUC__)
template<typename T>
static inline T assume(T v)
{
    if (!bool(v))
        __builtin_unreachable();
    return v;
}
#else
template<typename T>
static inline T assume(T v)
{
    return v;
}
#endif

static inline void _assume_not_none(PyObject *obj)
{
    assume(obj != Py_None);
}
#define assume_not_none(p) _assume_not_none((PyObject*)p)
    """
    T assume[T](T) noexcept nogil
    void assume_not_none(object) noexcept nogil
    void _assume_not_none "assume_not_none"(void*) noexcept nogil
