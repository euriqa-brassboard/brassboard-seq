/*************************************************************************
 *   Copyright (c) 2024 - 2025 Yichao Yu <yyc1992@gmail.com>             *
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

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <concepts>
#include <iostream>
#include <map>
#include <memory>
#include <ranges>
#include <span>
#include <vector>
#include <utility>

#include <stdint.h>
#include <stdio.h>

#ifdef __has_builtin
#  define bb_has_builtin(x) __has_builtin(x)
#else
#  define bb_has_builtin(x) 0
#endif

#define BB_CPU_X86_64 0
#define BB_CPU_X86 0
#define BB_CPU_AARCH64 0
#define BB_CPU_AARCH32 0
#define BB_CPU_PPC64 0
#define BB_CPU_PPC32 0

#if defined(__amd64__) || defined(__amd64) || defined(__x86_64__) || \
    defined(__x86_64) || defined(_M_X64) || defined(_M_AMD64)
#  undef BB_CPU_X86_64
#  define BB_CPU_X86_64 1
#elif defined(i386) || defined(__i386) || defined(__i386__) || defined(_M_IX86) || defined(_X86_)
#  undef BB_CPU_X86
#  define BB_CPU_X86 1
#elif defined(__aarch64__)
#  undef BB_CPU_AARCH64
#  define BB_CPU_AARCH64 1
#elif defined(__arm__) || defined(_M_ARM)
#  undef BB_CPU_AARCH32
#  define BB_CPU_AARCH32 1
#elif defined(__PPC64__)
#  undef BB_CPU_PPC64
#  define BB_CPU_PPC64 1
#elif defined(_ARCH_PPC)
#  undef BB_CPU_PPC32
#  define BB_CPU_PPC32 1
#endif

#if BB_CPU_X86 || BB_CPU_X86_64
#  include <immintrin.h>
#elif BB_CPU_AARCH64
#  include <arm_neon.h>
#endif

namespace {

__attribute__((always_inline, flatten))
static inline constexpr auto _pyx_find_base(auto *p, auto cb)
{
    if constexpr (requires { cb(p); }) {
        return p;
    }
    else {
        return _pyx_find_base(&p->__pyx_base, cb);
    }
}

}

#define pyx_find_base(p, fld)                                           \
    (::_pyx_find_base((p), [] (auto _x) constexpr requires requires { _x->fld; } {}))
#define pyx_fld(p, fld) (pyx_find_base((p), fld)->fld)

namespace brassboard_seq {

// Replace with C++23 [[assume()]];
#if bb_has_builtin(__builtin_assume)
static constexpr inline __attribute__((always_inline)) auto assume(auto v)
{
    __builtin_assume(bool(v));
    return v;
}
#elif defined(__GNUC__)
static constexpr inline __attribute__((always_inline)) auto assume(auto v)
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
static constexpr inline __attribute__((always_inline)) auto assume(auto v)
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
            oldval = std::exchange(other.oldval, (BacktraceTracker*)intptr_t(-1));
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
    return BacktraceTracker::GlobalRestorer(
        std::exchange(BacktraceTracker::global_tracker, tracker));
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

PyObject *py_catch_error(auto &&cb) try {
    return (PyObject*)cb();
}
catch (...) {
    return nullptr;
}

[[noreturn]] void py_num_arg_error(const char *func_name, ssize_t nfound,
                                   ssize_t nmin, ssize_t nmax);
static __attribute__((always_inline)) inline void
py_check_num_arg(const char *func_name, ssize_t nfound, ssize_t nmin, ssize_t nmax=-1)
{
    if ((nfound <= nmax || nmax < 0) && nfound >= nmin)
        return;
    py_num_arg_error(func_name, nfound, nmin, nmax);
}

static __attribute__((always_inline)) inline
bool get_value_bool(PyObject *obj, auto &&cb) requires requires { cb(); }
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
double get_value_f64(PyObject *obj, auto &&cb) requires requires { cb(); }
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
    Py_DECREF((PyObject*)std::exchange(field, py_newref(v)));
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

py_object channel_name_from_path(PyObject *path);

template<typename It>
struct py_iter {
    explicit py_iter(PyObject *obj) : obj(obj) {}
    auto begin() const { return It(obj); };
    auto end() const { return It::end(obj); }
private:
    PyObject *obj;
};

template<typename Value, typename Key>
struct _pydict_iterator {
    _pydict_iterator(PyObject *dict) : dict(dict)
    {
        ++(*this);
    }
    _pydict_iterator &operator++()
    {
        has_next = PyDict_Next(dict, &pos, &key, &value);
        return *this;
    }
    std::pair<Key*,Value*> operator*()
    {
        return { (Key*)key, (Value*)value };
    }
    bool operator==(std::nullptr_t) { return !has_next; }
    static std::nullptr_t end(auto) { return nullptr; };

private:
    PyObject *dict;
    PyObject *key, *value;
    Py_ssize_t pos{0};
    bool has_next;
};

template<typename Value>
struct _pylist_iterator {
    _pylist_iterator(PyObject *list) : list(list) {}
    _pylist_iterator &operator++()
    {
        ++pos;
        return *this;
    }
    std::pair<Py_ssize_t,Value*> operator*()
    {
        return { pos, (Value*)PyList_GET_ITEM(list, pos) };
    }
    bool operator==(Py_ssize_t n) { return pos == n; }
    static Py_ssize_t end(PyObject *list) { return PyList_GET_SIZE(list); }

private:
    PyObject *list;
    Py_ssize_t pos{0};
};

template<typename Value>
struct _pytuple_iterator {
    _pytuple_iterator(PyObject *tuple) : tuple(tuple) {}
    _pytuple_iterator &operator++()
    {
        ++pos;
        return *this;
    }
    std::pair<Py_ssize_t,Value*> operator*()
    {
        return { pos, (Value*)PyTuple_GET_ITEM(tuple, pos) };
    }
    bool operator==(Py_ssize_t n) { return pos == n; }
    static Py_ssize_t end(PyObject *tuple) { return PyTuple_GET_SIZE(tuple); }

private:
    PyObject *tuple;
    Py_ssize_t pos{0};
};

template<typename Value=PyObject, typename Key=PyObject>
using pydict_iter = py_iter<_pydict_iterator<Value,Key>>;
template<typename Value=PyObject>
using pylist_iter = py_iter<_pylist_iterator<Value>>;
template<typename Value=PyObject>
using pytuple_iter = py_iter<_pytuple_iterator<Value>>;

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

static inline bool py_issubtype_nontrivial(auto *a, auto *b)
{
    // Assume a != b and b != object, and skip the first and last element in mro.
    // Also assume fully initialized type a/b
    PyObject *mro = ((PyTypeObject*)a)->tp_mro;
    for (Py_ssize_t i = 1, n = PyTuple_GET_SIZE(mro) - 1; i < n; i++) {
        if (PyTuple_GET_ITEM(mro, i) == (PyObject*)b) {
            return true;
        }
    }
    return false;
}

void pytype_add_method(PyTypeObject *type, PyMethodDef *meth);
static inline void pytype_add_method(auto type, PyMethodDef *meth)
    requires (!std::same_as<std::remove_cvref_t<decltype(*type)>,PyTypeObject>)
{
    pytype_add_method((PyTypeObject*)type, meth);
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


// Assuming little endian
template<std::integral ELT,unsigned N>
struct Bits {
    static_assert(std::endian::native == std::endian::little);
    constexpr Bits() = default;
    constexpr Bits(const Bits&) = default;
    static constexpr Bits mask(unsigned b1, unsigned b2)
    {
        Bits res;
        if (b1 > b2)
            return res;
        unsigned idx1 = b1 / elbits;
        unsigned bit1 = b1 % elbits;
        unsigned idx2 = b2 / elbits;
        unsigned bit2 = b2 % elbits;
        if (idx1 == idx2) {
            res[idx1] = mask_ele(bit1, bit2);
        }
        else {
            res[idx1] = mask_ele(bit1, elbits - 1);
            for (unsigned i = idx1 + 1; i < idx2; i++)
                res[i] = ELT(-1);
            res[idx2] = mask_ele(0, bit2);
        }
        return res;
    }

    template<std::integral ELT2>
    explicit constexpr Bits(ELT2 v)
    {
        Bits<ELT2,1> res;
        res[0] = v;
        *this = Bits(res);
    }
    explicit constexpr Bits(std::span<ELT,N> data)
    {
        for (unsigned i = 0; i < N; i++) {
            bits[i] = data[i];
        }
    }

    constexpr operator bool() const
    {
        return std::ranges::any_of(bits, [] (auto v) { return bool(v); });
    }

    constexpr Bits operator<<(int n) const
    {
        if (n > 0)
            return left_shift(n);
        if (n < 0)
            return right_shift(-n);
        return *this;
    }
    constexpr Bits operator>>(int n) const
    {
        if (n > 0)
            return right_shift(n);
        if (n < 0)
            return left_shift(-n);
        return *this;
    }
    constexpr ELT &operator[](int n)
    {
        return bits[n];
    }
    constexpr const ELT &operator[](int n) const
    {
        return bits[n];
    }
    auto operator==(const Bits &other) const
    {
        return bits == other.bits;
    }
    auto operator<=>(const Bits &other) const
    {
        return std::lexicographical_compare_three_way(
            bits.rbegin(), bits.rend(), other.bits.rbegin(), other.bits.rend(),
            [&] (UELT v1, UELT v2) {
                return v1 <=> v2;
            });
    }

    template<std::integral ELT2,unsigned N2>
    constexpr auto &operator|=(const Bits<ELT2,N2> &other)
    {
        *this = *this | Bits(other);
        return *this;
    }
    template<std::integral ELT2,unsigned N2>
    constexpr auto &operator&=(const Bits<ELT2,N2> &other)
    {
        *this = *this & Bits(other);
        return *this;
    }
    template<std::integral ELT2,unsigned N2>
    constexpr auto &operator^=(const Bits<ELT2,N2> &other)
    {
        *this = *this ^ Bits(other);
        return *this;
    }

    constexpr auto operator~() const
    {
        Bits res;
        for (unsigned i = 0; i < N; i++)
            res.bits[i] = ~bits[i];
        return res;
    }
    template<std::integral ELT2,unsigned N2>
    constexpr auto operator|(const Bits<ELT2,N2> &other) const
    {
        return elwise_promote_op(other, [] (auto &a, auto &b) { a |= b; });
    }
    template<std::integral ELT2,unsigned N2>
    constexpr auto operator&(const Bits<ELT2,N2> &other) const
    {
        return elwise_promote_op(other, [] (auto &a, auto &b) { a &= b; });
    }
    template<std::integral ELT2,unsigned N2>
    constexpr auto operator^(const Bits<ELT2,N2> &other) const
    {
        return elwise_promote_op(other, [] (auto &a, auto &b) { a ^= b; });
    }
    template<std::integral ELT2,unsigned N2>
    constexpr operator Bits<ELT2,N2>() const
    {
        Bits<ELT2,N2> res;
        if constexpr (sizeof(ELT) == sizeof(ELT2)) {
            auto n = std::min(N, N2);
            for (unsigned i = 0; i < n; i++) {
                res.bits[i] = std::bit_cast<ELT2>(bits[i]);
            }
        }
        else if constexpr (sizeof(ELT) < sizeof(ELT2)) {
            // Cast to wider size
            static_assert(sizeof(ELT2) % sizeof(ELT) == 0);
            constexpr unsigned ne = sizeof(ELT2) / sizeof(ELT);
            auto n = std::min((N + ne - 1) / ne, N2);
            for (unsigned i = 0; i < n; i++) {
                std::array<ELT,ne> subbits{};
                for (unsigned j = 0; j < ne; j++) {
                    if (i * ne + j >= N)
                        break;
                    subbits[j] = bits[i * ne + j];
                }
                res.bits[i] = std::bit_cast<ELT2>(subbits);
            }
        }
        else {
            // Cast to narrower size
            static_assert(sizeof(ELT) % sizeof(ELT2) == 0);
            constexpr unsigned ne = sizeof(ELT) / sizeof(ELT2);
            constexpr auto _N = ne * N;
            auto _bits = std::bit_cast<std::array<ELT2,_N>>(bits);
            auto n = std::min(_N, N2);
            for (unsigned i = 0; i < n; i++) {
                res.bits[i] = std::bit_cast<ELT2>(_bits[i]);
            }
        }
        return res;
    }
    void print(std::ostream &stm) const
    {
        auto flags = stm.flags();
        stm.setf(std::ios_base::hex | std::ios_base::right,
                 std::ios_base::basefield | std::ios_base::adjustfield |
                 std::ios_base::showbase);
        auto fc = stm.fill();

        if (flags & std::ios_base::showbase) {
            stm.width(0);
            stm << "0x";
        }
        for (auto v: std::ranges::views::reverse(bits)) {
            stm.fill('0');
            stm.width(elbits / 4);
            if constexpr (elbits == 8) {
                stm << int(UELT(v));
            }
            else {
                stm << v;
            }
        }
        stm.fill(fc);
        stm.width(0);
        stm.setf(flags);
    }
    PyObject *to_pybytes() const
    {
        return throw_if_not(PyBytes_FromStringAndSize((const char*)&bits[0],
                                                      sizeof(bits)));
    }
    PyObject *to_pylong() const
    {
#if PY_VERSION_HEX >= 0x030d0000
        return throw_if_not(PyLong_FromUnsignedNativeBytes(&bits[0], sizeof(bits), 1));
#else
        return throw_if_not(_PyLong_FromByteArray((const unsigned char*)&bits[0],
                                                  sizeof(bits), true, 0));
#endif
    }

    std::array<ELT,N> bits{};
private:
    using UELT = std::make_unsigned_t<ELT>;
    static constexpr ELT mask_ele(unsigned bit1, unsigned bit2)
    {
        assert(bit1 < elbits);
        assert(bit2 < elbits);
        if (bit2 - bit1 == elbits - 1)
            return ELT(-1);
        UELT v = (UELT(1) << (bit2 - bit1 + 1)) - 1;
        return ELT(v << bit1);
    }
    template<std::integral ELT2,unsigned N2>
    constexpr auto elwise_promote_op(const Bits<ELT2,N2> &other, auto &&cb) const
    {
        if constexpr (sizeof(Bits) >= sizeof(Bits<ELT2,N2>)) {
            Bits res(other);
            for (unsigned i = 0; i < N; i++)
                cb(res.bits[i], bits[i]);
            return res;
        }
        else {
            Bits<ELT2,N2> res(*this);
            for (unsigned i = 0; i < N2; i++)
                cb(res.bits[i], other.bits[i]);
            return res;
        }
    }
    static constexpr unsigned elbits = sizeof(ELT) * 8;
    constexpr Bits left_shift(unsigned n) const
    {
        Bits res{};
        unsigned shift_idx = n / elbits;
        unsigned shift_bit = n % elbits;
        if (shift_idx >= N)
            return res;
        if (shift_bit == 0) {
            for (unsigned i = 0; i < N - shift_idx; i++)
                res.bits[i + shift_idx] = bits[i];
            return res;
        }
        unsigned rshift = elbits - shift_bit;
        res.bits[shift_idx] = ELT(UELT(bits[0]) << shift_bit);
        for (unsigned i = 1; i < N - shift_idx; i++)
            res.bits[i + shift_idx] = ELT((UELT(bits[i]) << shift_bit) |
                                          (UELT(bits[i - 1]) >> rshift));
        return res;
    }
    constexpr Bits right_shift(unsigned n) const
    {
        Bits res{};
        unsigned shift_idx = n / elbits;
        unsigned shift_bit = n % elbits;
        if (shift_idx >= N)
            return res;
        if (shift_bit == 0) {
            for (unsigned i = 0; i < N - shift_idx; i++)
                res.bits[i] = bits[i + shift_idx];
            return res;
        }
        unsigned lshift = elbits - shift_bit;
        for (unsigned i = 0; i < N - shift_idx - 1; i++)
            res.bits[i] = ELT((UELT(bits[i + shift_idx]) >> shift_bit) |
                              (UELT(bits[i + 1 + shift_idx]) << lshift));
        res.bits[N - shift_idx - 1] = ELT(UELT(bits[N - 1]) >> shift_bit);
        return res;
    }
    template<std::integral ELT2,unsigned N2> friend class Bits;
};

template<std::integral ELT,unsigned N>
static inline std::ostream &operator<<(std::ostream &io, const Bits<ELT,N> &bits)
{
    bits.print(io);
    return io;
}

// Similar to `std::stringstream` but is also indexable.
// The subtypes also support multiple different back storages
// and can generally transfer the ownership of the buffer
// which is not possible with `std::stringstream`.
class buff_streambuf : public std::streambuf {
public:
    buff_streambuf(size_t sz=0)
        : m_end(sz)
    {}
    pos_type tellg() const
    {
        return pptr() - pbase();
    }
    const char &operator[](size_t i) const
    {
        return pbase()[i];
    }
    char &operator[](size_t i)
    {
        return pbase()[i];
    }

private:
    std::streamsize xsputn(const char* s, std::streamsize count) override;
    int_type overflow(int_type ch) override;
    pos_type seekoff(off_type off, std::ios_base::seekdir dir,
                     std::ios_base::openmode which) override;
    pos_type seekpos(pos_type pos, std::ios_base::openmode which) override;
    int sync() override;

    pos_type _seekpos(pos_type pos);
    void update_size();

    // The base class defines most of the interface with the stream framework
    // and subclasses only need to define the `extend` method, which should
    // resize the buffer to fit at least `sz` bytes from the current pointer
    // without loosing any existing content (up to `m_end`).
    virtual char *extend(size_t sz) = 0;

protected:
    // This is the last location accessed on the stream.
    // In another word, this is the length of the file.
    ssize_t m_end;
};

class pybytes_streambuf : public buff_streambuf {
public:
    pybytes_streambuf();
    ~pybytes_streambuf() override;

    __attribute__((returns_nonnull)) PyObject *get_buf();

private:
    char *extend(size_t sz) override;

    PyObject *m_buf = nullptr;
};

class pybytearray_streambuf : public buff_streambuf {
public:
    pybytearray_streambuf();
    ~pybytearray_streambuf() override;

    __attribute__((returns_nonnull)) PyObject *get_buf();

private:
    char *extend(size_t sz) override;

    PyObject *m_buf = nullptr;
};

class buff_ostream : public std::ostream {
public:
    buff_ostream(buff_streambuf *buf)
        : std::ostream(buf)
    {}
    pos_type tellg()
    {
        return static_cast<buff_streambuf*>(rdbuf())->tellg();
    }
    const char &operator[](size_t i) const
    {
        return (*static_cast<const buff_streambuf*>(rdbuf()))[i];
    }
    char &operator[](size_t i)
    {
        return (*static_cast<buff_streambuf*>(rdbuf()))[i];
    }
};

class pybytes_ostream : public buff_ostream {
public:
    pybytes_ostream();
    ~pybytes_ostream();

    __attribute__((returns_nonnull)) PyObject *get_buf()
    {
        flush();
        return m_buf.get_buf();
    }

private:
    pybytes_streambuf m_buf;
};

class pybytearray_ostream : public buff_ostream {
public:
    pybytearray_ostream();
    ~pybytearray_ostream();

    __attribute__((returns_nonnull)) PyObject *get_buf()
    {
        flush();
        return m_buf.get_buf();
    }

private:
    pybytearray_streambuf m_buf;
};

template<std::signed_integral I, std::floating_point F>
static constexpr inline I round(F f)
{
    if (__builtin_constant_p(f))
        return I(f < 0 ? I(f - F(0.5)) : I(f + F(0.5)));
    constexpr auto Isz = sizeof(I);
    constexpr auto Ffloat = std::is_same_v<std::remove_cvref_t<F>,float>;
    constexpr auto Fdouble = std::is_same_v<std::remove_cvref_t<F>,double>;
#if BB_CPU_X86 || BB_CPU_X86_64
    if constexpr (Ffloat) {
        if constexpr (Isz <= 4) {
            return I(_mm_cvtss_si32(_mm_set_ss(f)));
        }
        else {
            return I(_mm_cvtss_si64(_mm_set_ss(f)));
        }
    }
    else if constexpr (Fdouble) {
        if constexpr (Isz <= 4) {
            return I(_mm_cvtsd_si32(_mm_set_sd(f)));
        }
        else {
            return I(_mm_cvtsd_si64(_mm_set_sd(f)));
        }
    }
#elif BB_CPU_AARCH64
    if constexpr (Ffloat) {
        return I(vcvtns_s32_f32(f));
    }
    else if constexpr (Fdouble) {
        return I(vcvtnd_s64_f64(f));
    }
#else
    if constexpr (false) {
    }
#endif
    else if constexpr (Isz > sizeof(long)) {
        return I(std::llrint(f));
    }
    else {
        return I(std::lrint(f));
    }
}

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

static void foreach_max_range(std::span<int> value, auto &&cb)
{
    int N = value.size();
    // Stack of v=>idx
    std::vector<std::pair<int,int>> maxv_stack;
    for (int i = 0; i < N; i++) {
        auto v = value[i];
        int start_idx = i;
        while (!maxv_stack.empty()) {
            auto [prev_v, prev_idx] = maxv_stack.back();
            if (prev_v < v)
                break;
            if (prev_v > v)
                cb(prev_idx, i - 1, prev_v);
            maxv_stack.pop_back();
            start_idx = prev_idx;
        }
        maxv_stack.emplace_back(v, start_idx);
    }
    for (auto [prev_v, prev_idx]: std::ranges::views::reverse(maxv_stack)) {
        cb(prev_idx, N - 1, prev_v);
    }
}

template<typename T>
static inline char *to_chars(std::span<char> buf, T &&t)
{
    auto [ptr, ec] = std::to_chars(buf.data(), buf.data() + buf.size(), t);
    if (ec != std::errc())
        throw std::system_error(std::make_error_code(ec));
    return ptr;
}

void init_library();

}

#endif
