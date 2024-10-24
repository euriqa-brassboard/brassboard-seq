/*************************************************************************
 *   Copyright (c) 2024 - 2024 Yichao Yu <yyc1992@gmail.com>             *
 *                                                                       *
 *   This library is free software; you can redistribute it and/or       *
 *   modify it under the terms of the GNU Lesser General Public          *
 *   License as published by the Free Software Foundation; either        *
 *   version 3.0 of the License, or (at your option) any later version.  *
 *                                                                       *
 *   This library is distributed in the hope that it will be useful,     *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of      *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU    *
 *   Lesser General Public License for more details.                     *
 *                                                                       *
 *   You should have received a copy of the GNU Lesser General Public    *
 *   License along with this library. If not,                            *
 *   see <http://www.gnu.org/licenses/>.                                 *
 *************************************************************************/

#ifndef BRASSBOARD_SEQ_SRC_UTILS_H
#define BRASSBOARD_SEQ_SRC_UTILS_H

#include "Python.h"
#include "frameobject.h"

#include <array>
#include <cmath>
#include <map>
#include <memory>
#include <span>
#include <vector>

#include <stdint.h>
#include <stdio.h>

#ifdef __has_builtin
#  define bb_has_builtin(x) __has_builtin(x)
#else
#  define bb_has_builtin(x) 0
#endif

namespace brassboard_seq {

// Replace with C++23 [[assume()]];
#if bb_has_builtin(__builtin_assume)
static inline __attribute__((always_inline)) auto assume(auto v)
{
    __builtin_assume(bool(v));
    return v;
}
#elif defined(__GNUC__)
static inline __attribute__((always_inline)) auto assume(auto v)
{
#  if __GNUC__ >= 13
    __attribute__((assume(bool(v))));
#  else
    if (!bool(v))
        __builtin_unreachable();
#  endif
    return v;
}
#else
static inline __attribute__((always_inline)) auto assume(auto v)
{
    return v;
}
#endif

static inline __attribute__((always_inline)) void assume_not_none(auto *obj)
{
    assume((PyObject*)obj != Py_None);
}

[[noreturn]] void throw0();

static inline __attribute__((always_inline)) auto throw_if_not(auto &&v)
{
    if (!v)
        throw0();
    return std::move(v);
}

static inline __attribute__((always_inline)) auto throw_if(auto &&v)
{
    if (v)
        throw0();
    return std::move(v);
}

enum BBLogLevel {
    BB_LOG_DEBUG,
    BB_LOG_INFO,
};
extern BBLogLevel bb_logging_level;

#define bb_log(level, ...) do {                 \
        if (bb_logging_level <= (level)) {      \
            printf(__VA_ARGS__);                \
        }                                       \
    } while (0)
#define bb_debug(...) bb_log(BB_LOG_DEBUG, __VA_ARGS__)
#define bb_info(...) bb_log(BB_LOG_INFO, __VA_ARGS__)

static inline auto py_newref(auto *obj)
{
    Py_INCREF(obj);
    return obj;
}

static inline auto py_xnewref(auto *obj)
{
    Py_XINCREF(obj);
    return obj;
}

#if PY_VERSION_HEX >= 0x030c0000
static inline auto py_immref(auto *obj)
{
    return obj;
}
#else
static inline auto py_immref(auto *obj)
{
    return py_newref(obj);
}
#endif

struct BacktraceTracker {
    // Record the backtrace to be used later.
    // We'd like to do this with the lowest overhead possible at record time.
    // Python backtrace is computed from the interpreter frame,
    // which forms a linked chain in the order of the call stack.
    // However, we cannot simply store the current frame itself for two reasons,
    // 1. The frame is mutable, the line number and lasti (last instruction?)
    //    pointer in the frame is updated as the python interpreter runs through
    //    a function. We have to get these numbers from the frame at recording time.
    // 2. The frame itself contains references to a lot of other objects
    //    that we don't want to keep around, this mostly includes the local variables
    //    that may even form reference cycles (if it refers to the python object
    //    we embed this structure in). Therefore we need to reference only objects
    //    that are intrinsically long lived.
    // Since we would like to construct a fake but proper python traceback object
    // we need to collect enough information to do that,
    // which means we need to reconstruct a fake frame
    // and record the lasti and lineno from the frame.
    struct FrameInfo {
        PyCodeObject *code;
        int lasti;
        int lineno;
        FrameInfo(PyFrameObject *frame);
        __attribute__((returns_nonnull)) PyObject *get_traceback(PyObject *next);
    };

    void _record(uintptr_t key);

    inline void record(uintptr_t key)
    {
        if (!max_frame)
            return;
        _record(key);
    }
    inline __attribute__((always_inline)) void record(void *key)
    {
        record((uintptr_t)key);
    }

    PyObject *get_backtrace(uintptr_t key);
    ~BacktraceTracker()
    {
        // Do the freeing here instead of in the destructor of the FrameInfo object
        // so that we don't need to worry about the FrameInfo object being
        // copied/moved around when we add the frames.
        for (auto &[key, trace]: traces) {
            for (auto &frame: trace) {
                Py_DECREF(frame.code);
            }
        }
    }

    static BacktraceTracker *global_tracker;
    struct GlobalRestorer {
        GlobalRestorer(BacktraceTracker *oldval)
            : oldval(oldval)
        {
        }
        GlobalRestorer() = default;
        GlobalRestorer(const GlobalRestorer&) = delete;
        GlobalRestorer(GlobalRestorer &&other)
            : oldval(other.oldval)
        {
            other.oldval = (BacktraceTracker*)intptr_t(-1);
        }
        GlobalRestorer &operator=(const GlobalRestorer&) = delete;
        GlobalRestorer &operator=(GlobalRestorer &&other)
        {
            oldval = other.oldval;
            other.oldval = (BacktraceTracker*)intptr_t(-1);
            return *this;
        }
        ~GlobalRestorer()
        {
            if (oldval != (BacktraceTracker*)intptr_t(-1)) {
                global_tracker = oldval;
            }
        }

        BacktraceTracker *oldval{(BacktraceTracker*)intptr_t(-1)};
    };

    int max_frame{0};
    std::map<uintptr_t,std::vector<FrameInfo>> traces;
};

static inline BacktraceTracker::GlobalRestorer
set_global_tracker(BacktraceTracker *tracker)
{
    auto oldval = BacktraceTracker::global_tracker;
    BacktraceTracker::global_tracker = tracker;
    return BacktraceTracker::GlobalRestorer(oldval);
}

static inline __attribute__((always_inline,pure))
uintptr_t event_time_key(void *event_time)
{
    return (uintptr_t)event_time;
}
static inline __attribute__((always_inline,pure))
uintptr_t action_key(int aid)
{
    return (uintptr_t)(aid << 2) | 1;
}
static inline __attribute__((always_inline,pure))
uintptr_t assert_key(int aid)
{
    return (uintptr_t)(aid << 2) | 2;
}

void bb_reraise(uintptr_t key);
void _bb_err_format(PyObject *exc, uintptr_t key, const char *format, ...);

[[noreturn]] void bb_rethrow(uintptr_t key);
[[noreturn]] void bb_throw_format(PyObject *exc, uintptr_t key,
                                  const char *format, ...);
[[noreturn]] void py_throw_format(PyObject *exc, const char *format, ...);

static inline __attribute__((always_inline)) auto throw_if_not(auto &&v, uintptr_t key)
{
    if (!v)
        bb_rethrow(key);
    return std::move(v);
}

static inline __attribute__((always_inline)) auto throw_if(auto &&v, uintptr_t key)
{
    if (v)
        bb_rethrow(key);
    return std::move(v);
}

// Wrapper inline function to make it more clear to the C compiler
// that the function returns 0
static inline __attribute__((always_inline))
int bb_err_format(PyObject *exc, uintptr_t key, const char *format, auto... args)
{
    _bb_err_format(exc, key, format, args...);
    return 0;
}
static inline __attribute__((always_inline))
PyObject *PyErr_Format(PyObject *exc, const char *format, auto... args)
{
    ::PyErr_Format(exc, format, args...);
    return nullptr;
}

static inline __attribute__((always_inline))
void bb_reraise_and_throw_if(bool cond, uintptr_t key)
{
    if (cond) {
        bb_rethrow(key);
    }
}

static __attribute__((always_inline)) inline
bool get_value_bool(PyObject *obj, auto &&cb)
{
    if (obj == Py_True)
        return true;
    if (obj == Py_False)
        return false;
    int res = PyObject_IsTrue(obj);
    if (res < 0)
        cb();
    return res;
}

static inline bool get_value_bool(PyObject *obj, uintptr_t key)
{
    return get_value_bool(obj, [&] {
        bb_rethrow(key);
    });
}

static __attribute__((always_inline)) inline
double get_value_f64(PyObject *obj, auto &&cb)
{
    if (PyFloat_CheckExact(obj)) [[likely]]
        return PyFloat_AS_DOUBLE(obj);
    auto res = PyFloat_AsDouble(obj);
    if (res == -1 && PyErr_Occurred())
        cb();
    return res;
}

static inline double get_value_f64(PyObject *obj, uintptr_t key)
{
    return get_value_f64(obj, [&] {
        bb_rethrow(key);
    });
}

struct PyDeleter {
    void operator()(auto *p) {
        if (p) {
            Py_DECREF(p);
        }
    }
};
struct py_object : std::unique_ptr<PyObject,PyDeleter> {
    using std::unique_ptr<PyObject,PyDeleter>::unique_ptr;
    operator PyObject*() { return this->get(); };
    void set_obj(PyObject *p)
    {
        reset(py_newref(p));
    }
};

extern PyObject *pyfloat_m1;
extern PyObject *pyfloat_m0_5;
extern PyObject *pyfloat_0;
extern PyObject *pyfloat_0_5;
extern PyObject *pyfloat_1;

static inline void pyassign(auto *&field, auto *v)
{
    auto oldval = field;
    field = py_newref(v);
    Py_DECREF((PyObject*)oldval);
}

__attribute__((returns_nonnull)) static inline PyObject*
pyfloat_from_double(double v)
{
    if (v == -1) {
        return py_newref(pyfloat_m1);
    }
    else if (v == -0.5) {
        return py_newref(pyfloat_m0_5);
    }
    else if (v == 0) {
        return py_newref(pyfloat_0);
    }
    else if (v == 0.5) {
        return py_newref(pyfloat_0_5);
    }
    else if (v == 1) {
        return py_newref(pyfloat_1);
    }
    return throw_if_not(PyFloat_FromDouble(v));
}

static constexpr int _pylong_cache_max = 4096;
extern const std::array<PyObject*,_pylong_cache_max * 2> _pylongs_cache;

__attribute__((returns_nonnull)) static inline PyObject*
pylong_cached(int v)
{
    assert(v < _pylong_cache_max);
    assert(v >= -_pylong_cache_max);
    return _pylongs_cache[v + _pylong_cache_max];
}

__attribute__((returns_nonnull)) static inline PyObject*
pylong_from_long(long v)
{
    if (v < _pylong_cache_max && v >= -_pylong_cache_max)
        return py_newref(pylong_cached(v));
    return throw_if_not(PyLong_FromLong(v));
}

__attribute__((returns_nonnull)) static inline PyObject*
pylong_from_longlong(long long v)
{
    if (v < _pylong_cache_max && v >= -_pylong_cache_max)
        return py_newref(pylong_cached(v));
    return throw_if_not(PyLong_FromLongLong(v));
}

__attribute__((returns_nonnull)) static inline PyObject*
pydict_new()
{
    return throw_if_not(PyDict_New());
}

__attribute__((returns_nonnull)) static inline PyObject*
pylist_new(Py_ssize_t n)
{
    return throw_if_not(PyList_New(n));
}

__attribute__((returns_nonnull)) static inline PyObject*
pytuple_new(Py_ssize_t n)
{
    return throw_if_not(PyTuple_New(n));
}

__attribute__((returns_nonnull)) static inline PyObject*
pyunicode_from_string(const char *str)
{
    return throw_if_not(PyUnicode_FromString(str));
}

template<typename T>
struct ValueIndexer {
    int get_id(void *p)
    {
        int nvalues = (int)values.size();
        auto [it, inserted] = indices.emplace(p, nvalues);
        if (inserted) {
            std::pair<void*,T> pair;
            pair.first = p;
            values.push_back(pair);
            return nvalues;
        }
        return it->second;
    }

    std::vector<std::pair<void*,T>> values;
    std::map<void*,int> indices;
};

template<typename CB>
struct ScopeExit {
    template<typename _CB>
    ScopeExit(_CB &&cb)
        : cb(std::forward<_CB>(cb))
    {}
    ~ScopeExit()
    {
        cb();
    }
private:
    CB cb;
};

template<typename CB>
ScopeExit(CB) -> ScopeExit<CB>;

static inline PyObject *pynum_add_or_sub(PyObject *a, PyObject *b, bool issub)
{
    if (issub) {
        return PyNumber_Subtract(a, b);
    }
    else {
        return PyNumber_Add(a, b);
    }
}

__attribute__((returns_nonnull))
PyObject *pytuple_append1(PyObject *tuple, PyObject *obj);
__attribute__((returns_nonnull)) PyObject *pydict_deepcopy(PyObject *d);

static inline void pylist_append(PyObject* list, PyObject* x)
{
    PyListObject *L = (PyListObject*)list;
    Py_ssize_t len = Py_SIZE(list);
    if (L->allocated > len && len > (L->allocated >> 1)) [[likely]] {
        Py_INCREF(x);
        PyList_SET_ITEM(list, len, x);
#if PY_VERSION_HEX >= 0x030900A4
        Py_SET_SIZE(list, len + 1);
#else
        Py_SIZE(list) = len + 1;
#endif
        return;
    }
    throw_if(PyList_Append(list, x));
}

__attribute__((returns_nonnull)) static inline PyObject*
pytype_genericalloc(auto *ty, Py_ssize_t sz=0)
{
    return throw_if_not(PyType_GenericAlloc((PyTypeObject*)ty, sz));
}

// Copied from cython
static inline PyObject* pyobject_call(PyObject *func, PyObject *arg,
                                      PyObject *kw=nullptr)
{
    auto call = Py_TYPE(func)->tp_call;
    if (!call) [[unlikely]]
        return PyObject_Call(func, arg, kw);
    if (Py_EnterRecursiveCall(" while calling a Python object")) [[unlikely]]
        return nullptr;
    auto result = call(func, arg, kw);
    Py_LeaveRecursiveCall();
    if (!result && !PyErr_Occurred()) [[unlikely]]
        PyErr_SetString(PyExc_SystemError,
                        "NULL result without error in PyObject_Call");
    return result;
}

static inline bool py_issubtype_nontrivial(PyTypeObject *a, PyTypeObject *b)
{
    // Assume a != b and b != object, and skip the first and last element in mro.
    // Also assume fully initialized type a/b
    PyObject *mro = a->tp_mro;
    for (Py_ssize_t i = 1, n = PyTuple_GET_SIZE(mro) - 1; i < n; i++) {
        if (PyTuple_GET_ITEM(mro, i) == (PyObject*)b) {
            return true;
        }
    }
    return false;
}

template<typename T, size_t N>
class PermAllocator {
public:
    template<typename ... Args>
    T *alloc(Args&&... args)
    {
        if (space_left == 0) {
            pages.push_back((T*)malloc(sizeof(T) * N));
            space_left = N;
        }
        auto page = pages.back();
        auto p = &page[N - space_left];
        space_left--;
        new (p) T(std::forward<Args>(args)...);
        return p;
    }
    PermAllocator() = default;
    PermAllocator(const PermAllocator&) = delete;
    PermAllocator(PermAllocator&&) = delete;

    ~PermAllocator()
    {
        auto npages = pages.size();
        for (size_t i = 0; i < npages; i++) {
            size_t cnt = i == npages - 1 ? N - space_left : N;
            auto page = pages[i];
            for (size_t j = 0; j < cnt; j++) {
                page[j].~T();
            }
            free(page);
        }
    }
private:
    std::vector<T*> pages;
    size_t space_left{0};
};

// Input: S
// Output: SA (require S.size() == SA.size())
// Character set must be within [0, N-1] where N is the size of S,
// Last character must be a unique 0.
// S will be overwritten.
void get_suffix_array(std::span<int> SA, std::span<int> S, std::span<int> ws);
static inline void order_to_rank(std::span<int> out, std::span<int> in)
{
    int N = in.size();
    for (int i = 0; i < N; i++) {
        out[in[i]] = i;
    }
}
void get_height_array(std::span<int> height, std::span<int> S,
                      std::span<int> SA, std::span<int> RK);

void init_library();

}

#endif
