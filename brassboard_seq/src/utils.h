//

#ifndef BRASSBOARD_SEQ_SRC_UTILS_H
#define BRASSBOARD_SEQ_SRC_UTILS_H

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

#endif
