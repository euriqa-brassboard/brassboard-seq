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

template<typename T>
__attribute__((always_inline, flatten))
static inline constexpr auto _pyx_find_base(T &&p, auto cb)
{
    if constexpr (requires { cb(std::forward<T>(p)); }) {
        if constexpr (std::is_pointer_v<std::remove_cvref_t<T>>) {
            return p;
        }
        else {
            return p.operator->();
        }
    }
    else {
        return _pyx_find_base(&p->__pyx_base, cb);
    }
}

}

#define pyx_find_base(p, fld)                                           \
    (::_pyx_find_base((p), [] (auto &&_x) constexpr requires requires { _x->fld; } {}))
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
static inline auto throw_if_not(auto &&v, uintptr_t key);

static inline __attribute__((always_inline)) auto throw_if(auto &&v)
{
    if (v)
        throw0();
    return std::move(v);
}
static inline auto throw_if(auto &&v, uintptr_t key);

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

struct CDeleter {
    template<typename T>
    void operator()(T *p) {
        free((void*)p);
    }
};

namespace py {

struct _common {};
template<typename T=PyObject> struct _ref;
template<typename T=PyObject> struct ptr;
template<template<typename> class H, typename T> struct common;
template<typename T=PyObject> using ref = _ref<T>;

template<typename T>
constexpr bool is_handle_v = std::is_base_of_v<_common,std::remove_cvref_t<T>>;

struct _pyobj_tag_type {};

struct _dict : _pyobj_tag_type {};
using dict = ptr<_dict>;
using dict_ref = ref<_dict>;
struct _set : _pyobj_tag_type {};
using set = ptr<_set>;
using set_ref = ref<_set>;
struct _list : _pyobj_tag_type {};
using list = ptr<_list>;
using list_ref = ref<_list>;
struct _tuple : _pyobj_tag_type {};
using tuple = ptr<_tuple>;
using tuple_ref = ref<_tuple>;
struct _bytes : _pyobj_tag_type {};
using bytes = ptr<_bytes>;
using bytes_ref = ref<_bytes>;
struct _float : _pyobj_tag_type {};
using float_ = ptr<_float>;
using float_ref = ref<_float>;
struct _int : _pyobj_tag_type {};
using int_ = ptr<_int>;
using int_ref = ref<_int>;
struct _str : _pyobj_tag_type {};
using str = ptr<_str>;
using str_ref = ref<_str>;
struct _mod : _pyobj_tag_type {};
using mod = ptr<_mod>;
using mod_ref = ref<_mod>;

template<typename T>
using py_ptr_type = std::conditional_t<std::is_base_of_v<_pyobj_tag_type,T>,PyObject,T>;

template<typename T>
struct _py_tag_type { using type = T; };
template<typename T>
struct _py_tag_type<ref<T>> { using type = T; };
template<typename T>
struct _py_tag_type<ptr<T>> { using type = T; };
template<typename T>
using py_tag_type = typename _py_tag_type<T>::type;

template<typename T2>
static constexpr bool is_py_ptr = is_handle_v<T2> ||
    (std::is_pointer_v<std::remove_cvref_t<T2>> &&
     !std::integral<std::remove_pointer_t<std::remove_cvref_t<T2>>>);

static inline void check_refcnt(auto *obj)
{
    assert(!obj || Py_REFCNT(obj) > 0);
}

static inline auto newref(auto *obj)
{
    check_refcnt(obj);
    Py_INCREF(obj);
    return obj;
}

static inline auto xnewref(auto *obj)
{
    check_refcnt(obj);
    Py_XINCREF(obj);
    return obj;
}

#if PY_VERSION_HEX >= 0x030c0000
static inline auto immref(auto *obj)
{
    return obj;
}
#else
static inline auto immref(auto *obj)
{
    check_refcnt(obj);
    return newref(obj);
}
#endif

template<typename T>
static inline py_ptr_type<T> *newref(py::ref<T> &&h)
{
    return h.rel();
}
template<template<typename> class H, typename T>
static inline py_ptr_type<T> *newref(const common<H,T> &h)
{
    return h.ref().rel();
}

template<template<typename> class H, typename T>
struct common : _common {
    static auto checked(auto *p, auto&&... args)
    {
        check_refcnt(p);
        return H<T>(throw_if_not(p, args...));
    }

    constexpr explicit operator bool() const noexcept
    {
        return _ptr() != nullptr;
    }
    constexpr auto operator->() const
    {
        return _ptr();
    }
    template<template<typename> class H2, typename T2>
    bool operator==(const common<H2,T2> &other) const
    {
        return (void*)_ptr() == (void*)other._ptr();
    }
    bool operator==(auto *ptr) const
    {
        return (void*)_ptr() == (void*)ptr;
    }
    bool is_none() const
    {
        return (void*)_ptr() == (void*)Py_None;
    }
    bool typeis(auto &&type) const
    {
        return Py_TYPE((PyObject*)_ptr()) == (PyTypeObject*)type;
    }

    template<typename T2=T>
    auto ptr() const
    {
        return py::ptr<T2>(_ptr());
    }
    template<typename T2=T>
    auto immref() const
    {
        return py::ref<T2>(py::immref(_ptr()));
    }
    template<typename T2=T>
    auto ref() const
    {
        return py::ref<T2>(py::newref(_ptr()));
    }
    template<typename T2=T>
    auto xref() const
    {
        return py::ref<T2>(py::xnewref(_ptr()));
    }

    auto str() const
    {
        return py::str_ref(throw_if_not(PyObject_Str((PyObject*)_ptr())));
    }
    auto list() const
    {
        return py::list_ref(throw_if_not(PySequence_List((PyObject*)_ptr())));
    }
    auto int_() const
    {
        return py::int_ref(throw_if_not(PyNumber_Long((PyObject*)_ptr())));
    }
    auto try_int() const;

    auto attr(const char *name) const
    {
        return py::ref(throw_if_not(PyObject_GetAttrString((PyObject*)_ptr(), name)));
    }
    template<typename T2>
    auto attr(T2 &&name) const requires is_py_ptr<T2>
    {
        return py::ref(throw_if_not(PyObject_GetAttr((PyObject*)_ptr(), (PyObject*)name)));
    }
    auto try_attr(const char *name) const
    {
        auto res = py::ref(PyObject_GetAttrString((PyObject*)_ptr(), name));
        if (!res)
            PyErr_Clear();
        return res;
    }
    template<typename T2>
    auto try_attr(T2 &&name) const requires is_py_ptr<T2>
    {
        auto res = py::ref(PyObject_GetAttr((PyObject*)_ptr(), (PyObject*)name));
        if (!res)
            PyErr_Clear();
        return res;
    }
    void set_attr(const char *name, auto &&val)
    {
        throw_if(PyObject_SetAttrString((PyObject*)_ptr(), name, (PyObject*)val));
    }
    template<typename T2>
    void set_attr(T2 &&name, auto &&val) requires is_py_ptr<T2>
    {
        throw_if(PyObject_SetAttr((PyObject*)_ptr(), (PyObject*)name, (PyObject*)val));
    }
    void del_attr(const char *name)
    {
        throw_if(PyObject_DelAttrString((PyObject*)_ptr(), name));
    }
    template<typename T2>
    void del_attr(T2 &&name) requires is_py_ptr<T2>
    {
        throw_if(PyObject_DelAttr((PyObject*)_ptr(), (PyObject*)name));
    }

    template<typename KW=PyObject*>
    auto vcall(PyObject *const *args, size_t nargsf, KW &&kwnames=nullptr)
    {
        return py::ref(throw_if_not(PyObject_Vectorcall((PyObject*)_ptr(), args,
                                                        nargsf, (PyObject*)kwnames)));
    }
    template<typename KW=PyObject*>
    auto vcall_dict(PyObject *const *args, size_t nargs, KW &&kws=nullptr)
    {
        return py::ref(throw_if_not(PyObject_VectorcallDict((PyObject*)_ptr(), args,
                                                            nargs, (PyObject*)kws)));
    }
    auto operator()(auto&&... args)
    {
        PyObject *py_args[] = { (PyObject*)args... };
        return vcall(py_args, sizeof...(args));
    }

    Py_ssize_t size() const requires std::same_as<T,_dict>
    {
        return PyDict_Size((PyObject*)_ptr());
    }
    template<typename Key>
    void set(Key &&key, auto &&val) requires (std::same_as<T,_dict> && is_py_ptr<Key>)
    {
        throw_if(PyDict_SetItem((PyObject*)_ptr(), (PyObject*)key, (PyObject*)val));
    }
    void set(const char *key, auto &&val) requires std::same_as<T,_dict>
    {
        throw_if(PyDict_SetItemString((PyObject*)_ptr(), key, (PyObject*)val));
    }
    auto try_get(auto &&key) const requires std::same_as<T,_dict>
    {
        auto res = py::ptr(PyDict_GetItemWithError((PyObject*)_ptr(), (PyObject*)key));
        if (!res)
            PyErr_Clear();
        return res;
    }
    bool contains(auto &&key) const requires std::same_as<T,_dict>
    {
        auto res = PyDict_Contains((PyObject*)_ptr(), (PyObject*)key);
        throw_if(res < 0);
        assume(res <= 1);
        return res;
    }
    void clear() requires std::same_as<T,_dict>
    {
        PyDict_Clear((PyObject*)_ptr());
    }
    auto copy() const requires std::same_as<T,_dict>
    {
        return dict_ref(throw_if_not(PyDict_Copy((PyObject*)_ptr())));
    }

    Py_ssize_t size() const requires std::same_as<T,_set>
    {
        return PySet_GET_SIZE((PyObject*)_ptr());
    }
    void add(auto &&item) requires std::same_as<T,_set>
    {
        throw_if(PySet_Add((PyObject*)_ptr(), (PyObject*)item));
    }
    bool contains(auto &&key) const requires std::same_as<T,_set>
    {
        auto res = PySet_Contains((PyObject*)_ptr(), (PyObject*)key);
        throw_if(res < 0);
        assume(res <= 1);
        return res;
    }
    void clear() requires std::same_as<T,_set>
    {
        throw_if(PySet_Clear((PyObject*)_ptr()));
    }

    Py_ssize_t size() const requires std::same_as<T,_list>
    {
        return PyList_GET_SIZE((PyObject*)_ptr());
    }
    template<typename T2>
    void SET(Py_ssize_t i, T2 &&val) requires std::same_as<T,_list>
    {
        PyList_SET_ITEM((PyObject*)_ptr(), i,
                        (PyObject*)py::newref(std::forward<T2>(val)));
    }
    void SET(Py_ssize_t i, std::nullptr_t) requires std::same_as<T,_list>
    {
        PyList_SET_ITEM((PyObject*)_ptr(), i, (PyObject*)nullptr);
    }
    template<typename T2>
    void set(Py_ssize_t i, T2 &&val) requires std::same_as<T,_list>
    {
        auto item = PyList_GET_ITEM((PyObject*)_ptr(), i);
        SET(i, std::forward<T2>(val));
        check_refcnt(item);
        Py_DECREF(item);
    }
    auto get(Py_ssize_t i) requires std::same_as<T,_list>
    {
        return py::ptr(PyList_GET_ITEM((PyObject*)_ptr(), i));
    }
    template<typename T2>
    void append(T2 &&x) requires std::same_as<T,_list>
    {
        auto list = (PyObject*)_ptr();
        PyListObject *L = (PyListObject*)list;
        Py_ssize_t len = Py_SIZE(list);
        if (L->allocated > len && len > (L->allocated >> 1)) [[likely]] {
            PyList_SET_ITEM(list, len, newref(std::forward<T2>(x)));
#if PY_VERSION_HEX >= 0x030900A4
            Py_SET_SIZE(list, len + 1);
#else
            Py_SIZE(list) = len + 1;
#endif
            return;
        }
        throw_if(PyList_Append(list, (PyObject*)x));
    }

    Py_ssize_t size() const requires std::same_as<T,_tuple>
    {
        return PyTuple_GET_SIZE((PyObject*)_ptr());
    }
    template<typename T2>
    void SET(Py_ssize_t i, T2 &&val) requires std::same_as<T,_tuple>
    {
        PyTuple_SET_ITEM((PyObject*)_ptr(), i,
                         (PyObject*)newref(std::forward<T2>(val)));
    }
    void SET(Py_ssize_t i, std::nullptr_t) requires std::same_as<T,_tuple>
    {
        PyTuple_SET_ITEM((PyObject*)_ptr(), i, (PyObject*)nullptr);
    }
    auto get(Py_ssize_t i) requires std::same_as<T,_tuple>
    {
        return py::ptr(PyTuple_GET_ITEM((PyObject*)_ptr(), i));
    }

    Py_ssize_t size() const requires std::same_as<T,_str>
    {
        return PyUnicode_GET_LENGTH((PyObject*)_ptr());
    }
    bool contains(auto &&key) const requires std::same_as<T,_str>
    {
        auto res = PyUnicode_Contains((PyObject*)_ptr(), (PyObject*)key);
        throw_if(res < 0);
        assume(res <= 1);
        return res;
    }
    auto concat(auto &&s2) const requires std::same_as<T,_str>
    {
        return str_ref(throw_if_not(PyUnicode_Concat((PyObject*)_ptr(), (PyObject*)s2)));
    }
    auto join(auto &&items) const requires std::same_as<T,_str>
    {
        return str_ref(throw_if_not(PyUnicode_Join((PyObject*)_ptr(), (PyObject*)items)));
    }

private:
    auto *_ptr() const
    {
        return static_cast<const H<T>*>(this)->get();
    }
    template<typename T2> friend class ptr;
    template<typename T2> friend class _ref;
};

template<typename T>
struct ptr : common<ptr,T> {
    constexpr ptr() = default;
    constexpr ptr(auto *ptr) : m_ptr{(T*)ptr}
    {
        check_refcnt(m_ptr);
    }
    template<template<typename> class H, typename T2>
    constexpr ptr(const common<H,T2> &h) : m_ptr{(T*)h._ptr()}
    {
        check_refcnt(m_ptr);
    }
    ptr &operator=(auto *p) noexcept
    {
        m_ptr = (T*)p;
        check_refcnt(m_ptr);
        return *this;
    }
    template<template<typename> class H, typename T2>
    ptr &operator=(const common<H,T2> &h) noexcept
    {
        m_ptr = (T*)h._ptr();
        check_refcnt(m_ptr);
        return *this;
    }
    template<typename T2> ptr &operator=(ref<T2>&&) noexcept = delete;
    using common<ptr,T>::get;
    template<typename T2=py_ptr_type<T>>
    constexpr T2 *get() const
    {
        check_refcnt(m_ptr);
        return (T2*)m_ptr;
    }
    template<typename T2> operator T2*() const
    {
        check_refcnt(m_ptr);
        return (T2*)m_ptr;
    }

private:
    T *m_ptr{nullptr};
    template<typename T2> friend class ptr;
    template<typename T2> friend class _ref;
};
template<typename T> ptr(T*) -> ptr<T>;

template<typename T>
struct _ref : common<_ref,T> {
    constexpr _ref() = default;
    // Take ownership
    explicit constexpr _ref(auto *ref) : m_ptr{(T*)ref}
    {
        check_refcnt(m_ptr);
    }
    constexpr _ref(_ref &&h) : m_ptr{h.m_ptr}
    {
        h.m_ptr = nullptr;
        check_refcnt(m_ptr);
    }
    template<typename T2>
    constexpr _ref(ref<T2> &&h) : m_ptr{(T*)h.m_ptr}
    {
        h.m_ptr = nullptr;
        check_refcnt(m_ptr);
    }
    _ref(const _ref&) = delete;
    template<typename T2> _ref(const ref<T2>&) = delete;
    template<typename T2> _ref(const ptr<T2>&) = delete;
    ~_ref()
    {
        check_refcnt(m_ptr);
        Py_XDECREF((PyObject*)m_ptr);
    }
    template<template<typename> class H, typename T2>
    _ref &operator=(const common<H,T2>&) = delete;
    _ref &operator=(const _ref&) = delete;
    template<typename T2>
    _ref &operator=(ref<T2> &&h) noexcept
    {
        take(std::move(h));
        return *this;
    }
    _ref &operator=(_ref &&h) noexcept
    {
        take(std::move(h));
        return *this;
    }
    using common<_ref,T>::get;
    template<typename T2=py_ptr_type<T>>
    constexpr T2 *get() const
    {
        check_refcnt(m_ptr);
        return (T2*)m_ptr;
    }
    template<typename T2>
    explicit operator T2*()
    {
        check_refcnt(m_ptr);
        return (T2*)m_ptr;
    }
    void take(auto *p) noexcept
    {
        check_refcnt(m_ptr);
        auto ptr = m_ptr;
        m_ptr = (T*)p;
        Py_XDECREF((PyObject*)ptr);
        check_refcnt(m_ptr);
    }
    void take_checked(auto *p, auto&&... args)
    {
        take(throw_if_not(p, args...));
    }
    template<typename T2>
    void take(ref<T2> &&h) noexcept
    {
        check_refcnt(m_ptr);
        auto ptr = m_ptr;
        m_ptr = (T*)h.rel();
        Py_XDECREF((PyObject*)ptr);
        check_refcnt(m_ptr);
    }
    template<typename T2=py_ptr_type<T>> T2 *rel()
    {
        auto p = m_ptr;
        m_ptr = nullptr;
        check_refcnt(p);
        return (T2*)p;
    }
    auto &_get_ptr_slot()
    {
        return m_ptr;
    }

private:
    T *m_ptr{nullptr};
    template<typename T2> friend class ptr;
    template<typename T2> friend class _ref;
};
template<typename T> _ref(T*) -> _ref<T>;

template<template<typename> class H, typename T>
inline auto common<H,T>::try_int() const
{
    auto res = py::int_ref(PyNumber_Long((PyObject*)_ptr()));
    if (!res)
        PyErr_Clear();
    return res;
}

template<typename T>
static inline void assign(auto *&field, T &&v)
{
    ref((PyObject*)std::exchange(field, py::newref(std::forward<T>(v))));
}

template<typename T>
static inline auto _vararg_decay(T &&v)
{
    if constexpr (is_handle_v<T>) {
        return (PyObject*)v;
    }
    else {
        return std::forward<T>(v);
    }
}

template<typename... Args>
static inline str_ref str_format(const char *format, Args&&... args)
{
    return str_ref::checked(
        PyUnicode_FromFormat(format, _vararg_decay(std::forward<Args>(args))...));
}

static inline dict_ref new_dict()
{
    return dict_ref(throw_if_not(PyDict_New()));
}
ref<> dict_deepcopy(ptr<> d);

template<typename T=PyObject*>
static inline set_ref new_set(T &&h=nullptr)
{
    return set_ref(throw_if_not(PySet_New((PyObject*)h)));
}

static inline list_ref new_list(Py_ssize_t n)
{
    return list_ref(throw_if_not(PyList_New(n)));
}

extern tuple empty_tuple;
static inline tuple_ref new_tuple(Py_ssize_t n)
{
    return tuple_ref(throw_if_not(PyTuple_New(n)));
}

extern bytes empty_bytes;
static inline bytes_ref new_bytes(const char *data, Py_ssize_t len)
{
    return ref(throw_if_not(PyBytes_FromStringAndSize(data, len)));
}

extern float_ float_m1;
extern float_ float_m0_5;
extern float_ float_0;
extern float_ float_0_5;
extern float_ float_1;
static inline float_ref new_float(double v)
{
    if (v == -1) {
        return float_m1.ref();
    }
    else if (v == -0.5) {
        return float_m0_5.ref();
    }
    else if (v == 0) {
        return float_0.ref();
    }
    else if (v == 0.5) {
        return float_0_5.ref();
    }
    else if (v == 1) {
        return float_1.ref();
    }
    return float_ref(throw_if_not(PyFloat_FromDouble(v)));
}

static constexpr int _int_cache_max = 4096;
extern const std::array<int_,_int_cache_max * 2> _int_cache;

template<std::integral auto v>
static consteval void assert_int_cache()
{
    static_assert(v < _int_cache_max);
    static_assert(v >= -_int_cache_max);
}

static inline int_ int_cached(int v)
{
    assert(v < _int_cache_max);
    assert(v >= -_int_cache_max);
    return _int_cache[v + _int_cache_max];
}

static inline int_ref new_int(std::integral auto v)
{
    if (v < _int_cache_max && v >= -_int_cache_max)
        return int_cached(v).ref();
    if constexpr (sizeof(v) > sizeof(long)) {
        return int_ref(throw_if_not(PyLong_FromLongLong(v)));
    }
    else {
        return int_ref(throw_if_not(PyLong_FromLong(v)));
    }
}

static inline str_ref new_str(const char *str)
{
    return str_ref(throw_if_not(PyUnicode_FromString(str)));
}
static inline str_ref new_str(const char *str, Py_ssize_t len)
{
    return str_ref(throw_if_not(PyUnicode_FromStringAndSize(str, len)));
}

static inline str_ref new_str(const std::string &str)
{
    return new_str(str.c_str(), str.size());
}

struct stringio {
    stringio &operator=(const stringio&) = delete;

    void write(str s);
    void write_str(ptr<> obj)
    {
        write(obj.str());
    }
    void write_ascii(const char *s, ssize_t len);
    void write_ascii(const char *s)
    {
        write_ascii(s, strlen(s));
    }
    void write_rep_ascii(int nrep, const char *s, ssize_t len);
    void write_rep_ascii(int nrep, const char *s)
    {
        write_rep_ascii(nrep, s, strlen(s));
    }
    std::pair<int,void*> reserve_buffer(int kind, ssize_t len);
    str_ref getvalue();

private:
    void write_kind(const void *data, int kind, ssize_t len);
    void check_size(size_t sz, int kind);

    std::unique_ptr<char,CDeleter> m_buff;
    size_t m_size{0};
    size_t m_pos{0};
    int m_kind{PyUnicode_1BYTE_KIND};
};

static inline mod_ref import_module(const char *str)
{
    return mod_ref(throw_if_not(PyImport_ImportModule(str)));
}

static inline mod_ref new_module(PyModuleDef *def)
{
    return mod_ref(throw_if_not(PyModule_Create(def)));
}

template<typename T1=PyObject*,typename T2=PyObject*>
static inline auto new_cfunc(PyMethodDef *ml, T1 &&self=nullptr, T2 &&mod=nullptr)
{
    return ref(throw_if_not(PyCFunction_NewEx(ml, (PyObject*)self, (PyObject*)mod)));
}

template<typename T=PyObject, typename Tty>
static inline auto generic_alloc(Tty &&ty, Py_ssize_t sz=0) requires is_py_ptr<Tty>
{
    return ref<py_tag_type<T>>::checked(PyType_GenericAlloc((PyTypeObject*)ty, sz));
}

template<typename T>
static inline auto generic_alloc(Py_ssize_t sz=0)
    requires requires { (PyTypeObject*)&T::Type; }
{
    return ref<T>::checked(PyType_GenericAlloc((PyTypeObject*)&T::Type, sz));
}

template<typename It, typename T>
struct _iter {
    explicit _iter(T &obj) : obj(obj) {}
    auto begin() { return It((PyObject*)obj); };
    auto end() { return It::end((PyObject*)obj); }
private:
    T &obj;
};

template<typename Value, typename Key>
struct _dict_iterator {
    _dict_iterator(PyObject *dict) : dict(dict)
    {
        ++(*this);
    }
    _dict_iterator &operator++()
    {
        has_next = PyDict_Next(dict, &pos, &key, &value);
        return *this;
    }
    std::pair<ptr<py::py_tag_type<Key>>,ptr<py::py_tag_type<Value>>> operator*()
    {
        return { ptr((py::py_tag_type<Key>*)key), ptr((py::py_tag_type<Value>*)value) };
    }
    bool operator==(std::nullptr_t) { return !has_next; }
    static std::nullptr_t end(auto&&) { return nullptr; };

private:
    PyObject *dict;
    PyObject *key, *value;
    Py_ssize_t pos{0};
    bool has_next;
};

template<typename Value=PyObject, typename Key=PyObject,typename T>
static inline auto dict_iter(T &&h)
{
    using _T = std::remove_cvref_t<T>;
    return _iter<_dict_iterator<Value,Key>,_T>(h);
}

template<typename Value>
struct _list_iterator {
    _list_iterator(PyObject *list) : list(list) {}
    _list_iterator &operator++()
    {
        ++pos;
        return *this;
    }
    std::pair<Py_ssize_t,ptr<py::py_tag_type<Value>>> operator*()
    {
        return { pos, ptr((py::py_tag_type<Value>*)PyList_GET_ITEM(list, pos)) };
    }
    bool operator==(Py_ssize_t n) { return pos == n; }
    static Py_ssize_t end(PyObject *list) { return PyList_GET_SIZE(list); }

private:
    PyObject *list;
    Py_ssize_t pos{0};
};

template<typename Value=PyObject,typename T>
static inline auto list_iter(T &&h)
{
    using _T = std::remove_cvref_t<T>;
    return _iter<_list_iterator<Value>,_T>(h);
}

template<typename Value>
struct _tuple_iterator {
    _tuple_iterator(PyObject *tuple) : tuple(tuple) {}
    _tuple_iterator &operator++()
    {
        ++pos;
        return *this;
    }
    std::pair<Py_ssize_t,ptr<py::py_tag_type<Value>>> operator*()
    {
        return { pos, ptr((py::py_tag_type<Value>*)PyTuple_GET_ITEM(tuple, pos)) };
    }
    bool operator==(Py_ssize_t n) { return pos == n; }
    static Py_ssize_t end(PyObject *tuple) { return PyTuple_GET_SIZE(tuple); }

private:
    PyObject *tuple;
    Py_ssize_t pos{0};
};

template<typename Value=PyObject,typename T>
static inline auto tuple_iter(T &&h)
{
    using _T = std::remove_cvref_t<T>;
    return _iter<_tuple_iterator<Value>,_T>(h);
}

struct _str_iterator {
    _str_iterator(PyObject *str)
        : data(PyUnicode_DATA(str)),
          kind(PyUnicode_KIND(str))
    {
    }
    _str_iterator &operator++()
    {
        ++idx;
        return *this;
    }
    std::pair<Py_ssize_t,Py_UCS4> operator*()
    {
        return { idx, PyUnicode_READ(kind, data, idx) };
    }
    bool operator==(Py_ssize_t n) { return idx == n; }
    static Py_ssize_t end(PyObject *str) { return PyUnicode_GET_LENGTH(str); }

private:
    void *data;
    int kind;
    Py_ssize_t idx{0};
};

template<typename T>
static inline auto str_iter(T &&h)
{
    using _T = std::remove_cvref_t<T>;
    return _iter<_str_iterator,_T>(h);
}

}

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

template<typename T, typename ...Args>
static inline void call_constructor(T *x, Args&&... args)
{
    new (x) T(std::forward<Args>(args)...);
}
template<typename T>
static inline void call_destructor(T *x)
{
    x->~T();
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

[[noreturn]] void bb_rethrow(uintptr_t key);
[[noreturn]] void _bb_throw_format(PyObject *exc, uintptr_t key,
                                   const char *format, ...);
[[noreturn]] void _py_throw_format(PyObject *exc, const char *format, ...);

template<typename... Args>
[[noreturn]] static inline void bb_throw_format(PyObject *exc, uintptr_t key,
                                                const char *format, Args&&... args)
{
    _bb_throw_format(exc, key, format, py::_vararg_decay(std::forward<Args>(args))...);
}

template<typename... Args>
[[noreturn]] static inline void py_throw_format(PyObject *exc, const char *format,
                                                Args&&... args)
{
    _py_throw_format(exc, format, py::_vararg_decay(std::forward<Args>(args))...);
}

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
template<typename... Args>
static inline __attribute__((always_inline))
PyObject *PyErr_Format(PyObject *exc, const char *format, Args&&... args)
{
    ::PyErr_Format(exc, format, py::_vararg_decay(std::forward<Args>(args))...);
    return nullptr;
}

static inline __attribute__((always_inline))
void bb_rethrow_if(bool cond, uintptr_t key)
{
    if (cond) {
        bb_rethrow(key);
    }
}

void handle_cxx_exception();

auto cxx_catch(auto &&cb)
{
    using raw_ret_type = std::remove_cvref_t<decltype(cb())>;
    constexpr bool is_ptr = std::is_pointer_v<raw_ret_type>;
    constexpr bool is_handle = py::is_handle_v<raw_ret_type>;
    constexpr bool is_ref = is_handle && requires { cb().rel(); };
    try {
        if constexpr (is_ref) {
            return cb().rel();
        }
        else if constexpr (is_handle) {
            return cb().get();
        }
        else if constexpr (is_ptr) {
            return cb();
        }
        else {
            return (int)cb();
        }
    }
    catch (...) {
        handle_cxx_exception();
        if constexpr (is_ptr) {
            return (raw_ret_type)nullptr;
        }
        else if constexpr (is_handle) {
            using ret_type = std::remove_cvref_t<decltype(cb().get())>;
            return (ret_type)nullptr;
        }
        else {
            return -1;
        }
    }
}

static inline void throw_pyerr(bool cond=true)
{
    throw_if(cond && PyErr_Occurred());
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
    template<typename T> operator T*() { return (T*)this->get(); };
    void reset_checked(PyObject *obj, auto&&... args)
    {
        reset(throw_if_not(obj, args...));
    }
    void set_obj(PyObject *p)
    {
        reset(py::newref(p));
    }
};

__attribute__((returns_nonnull)) static inline PyObject*
pyfloat_from_double(double v)
{
    // Used by cython
    return py::new_float(v).rel();
}

__attribute__((returns_nonnull)) static inline PyObject*
pylong_from_long(long v)
{
    // Used by cython
    return py::new_int(v).rel();
}

__attribute__((returns_nonnull)) static inline PyObject*
pylong_from_longlong(long long v)
{
    // Used by cython
    return py::new_int(v).rel();
}

template<size_t N>
struct str_literal {
    constexpr str_literal(const char (&str)[N])
    {
        std::copy_n(str, N, value);
    }
    char value[N];
};

template<str_literal lit>
static const py::str _py_string_cache = py::str(py::new_str(lit.value).rel());
template<str_literal lit>
static inline auto operator ""_py()
{
    return _py_string_cache<lit>;
}

template<str_literal lit>
static inline auto operator ""_pymod()
{
    // Use a local static variable to make sure the initialization order
    // is correct when this is used to initialize another global variable
    static auto m = py::mod(py::import_module(lit.value).rel());
    return m;
}

struct py_stringio: py::stringio {
    void write(PyObject *str)
    {
        return py::stringio::write(str);
    }
    void write_str(PyObject *obj)
    {
        return py::stringio::write_str(obj);
    }
    __attribute__((returns_nonnull)) PyObject *getvalue()
    {
        return py::stringio::getvalue().rel();
    }
};

py::str_ref channel_name_from_path(PyObject *path);

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

__attribute__((returns_nonnull))
PyObject *pytuple_append1(PyObject *tuple, PyObject *obj);
static inline PyObject *pydict_deepcopy(PyObject *d)
{
    // Used by cython
    return py::dict_deepcopy(d).rel();
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
        return py::new_bytes((const char*)&bits[0], sizeof(bits)).rel();
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

    py::ref<> m_buf;
};

class pybytearray_streambuf : public buff_streambuf {
public:
    pybytearray_streambuf();
    ~pybytearray_streambuf() override;

    __attribute__((returns_nonnull)) PyObject *get_buf();

private:
    char *extend(size_t sz) override;

    py::ref<> m_buf;
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

}

#endif
