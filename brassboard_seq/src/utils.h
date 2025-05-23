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

#include <Python.h>
#include <frameobject.h>
#include <structmember.h>

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <concepts>
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

#ifdef __clang__
#  define OPT_FAST_MATH
#elif defined(__GNUC__)
#  define OPT_FAST_MATH __attribute__((optimize("-ffast-math")))
#else
#  define OPT_FAST_MATH
#endif

[[noreturn]] void throw0();
[[noreturn]] void bb_rethrow(uintptr_t key);

static inline __attribute__((always_inline)) auto throw_if_not(auto &&v)
{
    if (!v)
        throw0();
    return std::move(v);
}
static inline __attribute__((always_inline)) auto throw_if_not(auto &&v, uintptr_t key)
{
    if (!v)
        bb_rethrow(key);
    return std::move(v);
}

static inline __attribute__((always_inline)) auto throw_if(auto &&v)
{
    if (v)
        throw0();
    return std::move(v);
}
static inline __attribute__((always_inline)) auto throw_if(auto &&v, uintptr_t key)
{
    if (v)
        bb_rethrow(key);
    return std::move(v);
}

template<typename... Args>
[[noreturn]] static inline void py_throw_format(PyObject *exc, const char *format,
                                                Args&&... args);

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

template<size_t N>
struct str_literal {
    constexpr str_literal(const char (&str)[N])
    {
        std::copy_n(str, N, value);
    }
    constexpr str_literal()
    {
        std::fill_n(value, N, 0);
    }
    template<size_t N2>
    constexpr auto operator+(const str_literal<N2> &s2) const
    {
        str_literal<N + N2 - 1> res;
        std::copy_n(value, N - 1, res.value);
        std::copy_n(s2.value, N2, res.value + N - 1);
        return res;
    }
    template<size_t N2>
    constexpr auto operator+(const char (&s2)[N2]) const
    {
        return this->operator+(str_literal<N2>(s2));
    }
    constexpr operator const char*() const
    {
        return value;
    }
    char value[N];
};

template<typename... Args>
static inline char *to_chars(std::span<char> buf, Args&&... args)
{
    auto [ptr, ec] = std::to_chars(buf.data(), buf.data() + buf.size(),
                                   std::forward<Args>(args)...);
    if (ec != std::errc())
        throw std::system_error(std::make_error_code(ec));
    return ptr;
}

template<typename T, auto... fld> struct _field_pack {};
template<typename T, auto... fld>
struct __field_pack { using type = _field_pack<T,fld...>; };
template<typename T, auto... fld, typename T2, typename R, R T2::*fld1, auto... fld2>
struct __field_pack<_field_pack<T,fld...>,fld1,fld2...> {
    using type = _field_pack<T2,fld...,fld1,fld2...>;
};
template<typename T, auto... fld> using field_pack = typename __field_pack<T,fld...>::type;

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

namespace py {

struct _common {};
template<typename T=PyObject> struct _ref;
template<typename T> _ref(T*) -> _ref<T>;
template<typename T=PyObject> struct ptr;
template<typename T> ptr(T*) -> ptr<T>;
template<typename T> ptr(_ref<T>&) -> ptr<T>;
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
struct _bytearray : _pyobj_tag_type {};
using bytearray = ptr<_bytearray>;
using bytearray_ref = ref<_bytearray>;
struct _float : _pyobj_tag_type {};
using float_ = ptr<_float>;
using float_ref = ref<_float>;
struct _int : _pyobj_tag_type {};
using int_ = ptr<_int>;
using int_ref = ref<_int>;
struct _bool : _pyobj_tag_type {};
using bool_ = ptr<_bool>;
using bool_ref = ref<_bool>;
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

static constexpr void check_refcnt(auto *obj)
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
static inline auto *newref(ref<T> &&h)
{
    return h.rel();
}
template<template<typename> class H, typename T>
static inline auto *newref(const common<H,T> &h)
{
    return h.ref().rel();
}

static inline void DECREF(auto *obj)
{
    check_refcnt(obj);
    Py_DECREF((PyObject*)obj);
}

static inline void CLEAR(auto *&ptr_ref)
{
    auto obj = (PyObject*)ptr_ref;
    if (obj) {
        ptr_ref = nullptr;
        DECREF(obj);
    }
}

static inline void XDECREF(auto *obj)
{
    CLEAR(obj);
}

template<template<typename> class H, typename T>
struct common : _common {
private:
    template<typename T2=void, typename T3>
    static auto __ref(T3 *v);
    template<typename T2=void, typename T3>
    static auto __ptr(T3 *v);

public:
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
    auto type() const
    {
        return __ptr(Py_TYPE((PyObject*)_ptr()));
    }
    bool isinstance(auto &&type) const
    {
        auto res = PyObject_IsInstance((PyObject*)_ptr(), (PyObject*)type);
        throw_if(res < 0);
        assume(res <= 1);
        return res;
    }
    template<bool exact=false>
    bool isa(auto &&type) const
    {
        if constexpr (exact) {
            return this->type() == (PyTypeObject*)type;
        }
        else {
            return PyObject_TypeCheck((PyObject*)_ptr(), (PyTypeObject*)type);
        }
    }
    template<typename T2, bool exact=false> bool isa() const
        requires requires { (PyTypeObject*)&T2::Type; }
    {
        return isa<exact>((PyTypeObject*)&T2::Type);
    }
    template<typename T2, bool exact=false> bool isa() const
        requires std::same_as<py_tag_type<T2>,_dict>
    {
        if constexpr (exact) {
            return PyDict_CheckExact((PyObject*)_ptr());
        }
        else {
            return PyDict_Check((PyObject*)_ptr());
        }
    }
    template<typename T2, bool exact=false> bool isa() const
        requires std::same_as<py_tag_type<T2>,_set>
    {
        if constexpr (exact) {
#if PY_VERSION_HEX >= 0x030a0000
            return PySet_CheckExact((PyObject*)_ptr());
#else
            return type() == &PySet_Type;
#endif
        }
        else {
            return PySet_Check((PyObject*)_ptr());
        }
    }
    template<typename T2, bool exact=false> bool isa() const
        requires std::same_as<py_tag_type<T2>,_list>
    {
        if constexpr (exact) {
            return PyList_CheckExact((PyObject*)_ptr());
        }
        else {
            return PyList_Check((PyObject*)_ptr());
        }
    }
    template<typename T2, bool exact=false> bool isa() const
        requires std::same_as<py_tag_type<T2>,_tuple>
    {
        if constexpr (exact) {
            return PyTuple_CheckExact((PyObject*)_ptr());
        }
        else {
            return PyTuple_Check((PyObject*)_ptr());
        }
    }
    template<typename T2, bool exact=false> bool isa() const
        requires std::same_as<py_tag_type<T2>,_bytes>
    {
        if constexpr (exact) {
            return PyBytes_CheckExact((PyObject*)_ptr());
        }
        else {
            return PyBytes_Check((PyObject*)_ptr());
        }
    }
    template<typename T2, bool exact=false> bool isa() const
        requires std::same_as<py_tag_type<T2>,_bytearray>
    {
        if constexpr (exact) {
            return PyByteArray_CheckExact((PyObject*)_ptr());
        }
        else {
            return PyByteArray_Check((PyObject*)_ptr());
        }
    }
    template<typename T2, bool exact=false> bool isa() const
        requires std::same_as<py_tag_type<T2>,_float>
    {
        if constexpr (exact) {
            return PyFloat_CheckExact((PyObject*)_ptr());
        }
        else {
            return PyFloat_Check((PyObject*)_ptr());
        }
    }
    template<typename T2, bool exact=false> bool isa() const
        requires std::same_as<py_tag_type<T2>,_int>
    {
        if constexpr (exact) {
            return PyLong_CheckExact((PyObject*)_ptr());
        }
        else {
            return PyLong_Check((PyObject*)_ptr());
        }
    }
    template<typename T2, bool exact=false> bool isa() const
        requires std::same_as<py_tag_type<T2>,_bool>
    {
        if constexpr (exact) {
            auto obj = (PyObject*)_ptr();
            return obj == Py_True || obj == Py_False;
        }
        else {
            return PyBool_Check((PyObject*)_ptr());
        }
    }
    template<typename T2, bool exact=false> bool isa() const
        requires std::same_as<py_tag_type<T2>,_str>
    {
        if constexpr (exact) {
            return PyUnicode_CheckExact((PyObject*)_ptr());
        }
        else {
            return PyUnicode_Check((PyObject*)_ptr());
        }
    }
    template<typename T2, bool exact=false> bool isa() const
        requires std::same_as<py_tag_type<T2>,_mod>
    {
        if constexpr (exact) {
            return PyModule_CheckExact((PyObject*)_ptr());
        }
        else {
            return PyModule_Check((PyObject*)_ptr());
        }
    }
    template<typename T2>
    bool typeis(T2 &&type) const
    {
        return isa<true>(std::forward<T2>(type));
    }
    template<typename T2> bool typeis() const
    {
        return isa<T2,true>();
    }

    template<typename T2=T>
    auto ptr() const
    {
        return __ptr<py_tag_type<T2>>(_ptr());
    }
    template<typename T2=T>
    auto immref() const
    {
        return __ref<py_tag_type<T2>>(py::immref(_ptr()));
    }
    template<typename T2=T>
    auto ref() const
    {
        return __ref<py_tag_type<T2>>(newref(_ptr()));
    }
    template<typename T2=T>
    auto xref() const
    {
        return __ref<py_tag_type<T2>>(xnewref(_ptr()));
    }

    auto str() const
    {
        return str_ref(throw_if_not(PyObject_Str((PyObject*)_ptr())));
    }
    auto list() const
    {
        return list_ref(throw_if_not(PySequence_List((PyObject*)_ptr())));
    }
    auto int_() const
    {
        return int_ref(throw_if_not(PyNumber_Long((PyObject*)_ptr())));
    }
    auto try_int() const
    {
        auto res = __ref<_int>(PyNumber_Long((PyObject*)_ptr()));
        if (!res)
            PyErr_Clear();
        return res;
    }
    bool as_bool(auto &&cb) const requires requires { cb(); }
    {
        auto obj = (PyObject*)_ptr();
        if (obj == Py_True)
            return true;
        if (obj == Py_False) [[likely]]
            return false;
        int res = PyObject_IsTrue(obj);
        if (res < 0)
            cb();
        assume(res <= 1);
        return res;
    }
    bool as_bool(uintptr_t key=-1) const
    {
        return as_bool([&] { bb_rethrow(key); });
    }
    template<std::integral Ti=long>
    Ti as_int(auto &&cb) const requires requires { cb(); }
    {
        auto obj = (PyObject*)_ptr();
        Ti res;
        if constexpr (sizeof(Ti) > sizeof(long)) {
            static_assert(sizeof(Ti) <= sizeof(long long));
            res = (std::is_signed_v<Ti> ? PyLong_AsLongLong(obj) :
                   PyLong_AsUnsignedLongLong(obj));
        }
        else {
            res = (std::is_signed_v<Ti> ? PyLong_AsLong(obj) :
                   PyLong_AsUnsignedLong(obj));
        }
        if (res == (Ti)-1 && PyErr_Occurred())
            cb();
        return res;
    }
    template<std::integral Ti=long>
    Ti as_int(uintptr_t key=-1) const
    {
        return as_int<Ti>([&] { bb_rethrow(key); });
    }
    double as_float(auto &&cb) const requires requires { cb(); }
    {
        auto obj = (PyObject*)_ptr();
        if (std::same_as<T,_float> || isa<float_>())
            return PyFloat_AS_DOUBLE(obj);
        auto res = PyFloat_AsDouble(obj);
        if (res == -1 && PyErr_Occurred())
            cb();
        return res;
    }
    double as_float(uintptr_t key=-1) const
    {
        return as_float([&] { bb_rethrow(key); });
    }

    auto attr(const char *name) const
    {
        return __ref(throw_if_not(PyObject_GetAttrString((PyObject*)_ptr(), name)));
    }
    template<typename T2>
    auto attr(T2 &&name) const requires is_py_ptr<T2>
    {
        return __ref(throw_if_not(PyObject_GetAttr((PyObject*)_ptr(), (PyObject*)name)));
    }
    auto try_attr(const char *name) const
    {
        auto res = __ref(PyObject_GetAttrString((PyObject*)_ptr(), name));
        if (!res)
            PyErr_Clear();
        return res;
    }
    template<typename T2>
    auto try_attr(T2 &&name) const requires is_py_ptr<T2>
    {
        auto res = __ref(PyObject_GetAttr((PyObject*)_ptr(), (PyObject*)name));
        if (!res)
            PyErr_Clear();
        return res;
    }
    void set_attr(const char *name, auto &&val) const
    {
        throw_if(PyObject_SetAttrString((PyObject*)_ptr(), name, (PyObject*)val));
    }
    template<typename T2>
    void set_attr(T2 &&name, auto &&val) const requires is_py_ptr<T2>
    {
        throw_if(PyObject_SetAttr((PyObject*)_ptr(), (PyObject*)name, (PyObject*)val));
    }
    void del_attr(const char *name) const
    {
        throw_if(PyObject_DelAttrString((PyObject*)_ptr(), name));
    }
    template<typename T2>
    void del_attr(T2 &&name) const requires is_py_ptr<T2>
    {
        throw_if(PyObject_DelAttr((PyObject*)_ptr(), (PyObject*)name));
    }
    template<typename T2>
    void setitem(Py_ssize_t i, T2 &&val) const
    {
        throw_if(PySequence_SetItem((PyObject*)_ptr(), i, (PyObject*)val));
    }
    template<typename T2=PyObject>
    auto getitem(Py_ssize_t i) const
    {
        return __ref<py_tag_type<T2>>(throw_if_not(PySequence_GetItem((PyObject*)_ptr(), i)));
    }
    auto length() const
    {
        auto len = PySequence_Length((PyObject*)_ptr());
        throw_if(len < 0);
        return len;
    }

    template<typename KW=PyObject*>
    auto vcall(PyObject *const *args, size_t nargsf, KW &&kwnames=nullptr) const
    {
        return __ref(throw_if_not(PyObject_Vectorcall((PyObject*)_ptr(), args,
                                                      nargsf, (PyObject*)kwnames)));
    }
    auto operator()(auto&&... args) const
    {
        PyObject *py_args[] = { (PyObject*)args... };
        return vcall(py_args, sizeof...(args));
    }
    auto operator+(auto &&o) const
    {
        return __ref(throw_if_not(PyNumber_Add((PyObject*)_ptr(), (PyObject*)o)));
    }
    auto operator-(auto &&o) const
    {
        return __ref(throw_if_not(PyNumber_Subtract((PyObject*)_ptr(), (PyObject*)o)));
    }
    auto operator*(auto &&o) const
    {
        return __ref(throw_if_not(PyNumber_Multiply((PyObject*)_ptr(), (PyObject*)o)));
    }

    template<typename Value=PyObject> auto generic_iter(uintptr_t key=-1) const;

    Py_ssize_t size() const requires std::same_as<T,_dict>
    {
        return PyDict_GET_SIZE((PyObject*)_ptr());
    }
    template<typename Key>
    void set(Key &&key, auto &&val) const requires (std::same_as<T,_dict> && is_py_ptr<Key>)
    {
        throw_if(PyDict_SetItem((PyObject*)_ptr(), (PyObject*)key, (PyObject*)val));
    }
    void set(const char *key, auto &&val) const requires std::same_as<T,_dict>
    {
        throw_if(PyDict_SetItemString((PyObject*)_ptr(), key, (PyObject*)val));
    }
    template<typename T2=PyObject>
    auto try_get(auto &&key) const requires std::same_as<T,_dict>
    {
        auto res = __ptr<py_tag_type<T2>>(PyDict_GetItemWithError((PyObject*)_ptr(),
                                                                  (PyObject*)key));
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
    void clear() const requires std::same_as<T,_dict>
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
    void add(auto &&item) const requires std::same_as<T,_set>
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
    void clear() const requires std::same_as<T,_set>
    {
        throw_if(PySet_Clear((PyObject*)_ptr()));
    }

    Py_ssize_t size() const requires std::same_as<T,_list>
    {
        return PyList_GET_SIZE((PyObject*)_ptr());
    }
    template<typename T2>
    void SET(Py_ssize_t i, T2 &&val) const requires std::same_as<T,_list>
    {
        PyList_SET_ITEM((PyObject*)_ptr(), i,
                        (PyObject*)newref(std::forward<T2>(val)));
    }
    void SET(Py_ssize_t i, std::nullptr_t) const requires std::same_as<T,_list>
    {
        PyList_SET_ITEM((PyObject*)_ptr(), i, (PyObject*)nullptr);
    }
    template<typename T2>
    void set(Py_ssize_t i, T2 &&val) const requires std::same_as<T,_list>
    {
        auto item = PyList_GET_ITEM((PyObject*)_ptr(), i);
        SET(i, std::forward<T2>(val));
        DECREF(item);
    }
    template<typename T2=PyObject>
    auto get(Py_ssize_t i) const requires std::same_as<T,_list>
    {
        return __ptr<py_tag_type<T2>>(PyList_GET_ITEM((PyObject*)_ptr(), i));
    }
    template<typename T2>
    void append(T2 &&x) const requires std::same_as<T,_list>
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
    void SET(Py_ssize_t i, T2 &&val) const requires std::same_as<T,_tuple>
    {
        PyTuple_SET_ITEM((PyObject*)_ptr(), i,
                         (PyObject*)newref(std::forward<T2>(val)));
    }
    void SET(Py_ssize_t i, std::nullptr_t) const requires std::same_as<T,_tuple>
    {
        PyTuple_SET_ITEM((PyObject*)_ptr(), i, (PyObject*)nullptr);
    }
    template<typename T2=PyObject>
    auto get(Py_ssize_t i) const requires std::same_as<T,_tuple>
    {
        return __ptr<py_tag_type<T2>>(PyTuple_GET_ITEM((PyObject*)_ptr(), i));
    }
    template<typename T2> auto append(T2 &&v) const requires std::same_as<T,_tuple>;

    Py_ssize_t size() const requires std::same_as<T,_bytes>
    {
        return PyBytes_GET_SIZE((PyObject*)_ptr());
    }
    char *data() const requires std::same_as<T,_bytes>
    {
        return PyBytes_AS_STRING((PyObject*)_ptr());
    }
    auto decode() const requires std::same_as<T,_bytes>
    {
        return str_ref(throw_if_not(PyUnicode_DecodeUTF8(data(), size(), nullptr)));
    }

    Py_ssize_t size() const requires std::same_as<T,_bytearray>
    {
        return PyByteArray_GET_SIZE((PyObject*)_ptr());
    }
    char *data() const requires std::same_as<T,_bytearray>
    {
        return PyByteArray_AS_STRING((PyObject*)_ptr());
    }
    void resize(Py_ssize_t sz) const requires std::same_as<T,_bytearray>
    {
        throw_if(PyByteArray_Resize((PyObject*)_ptr(), sz));
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
    int compare(auto &&other) const requires std::same_as<T,_str>
    {
        auto res = PyUnicode_Compare((PyObject*)_ptr(), (PyObject*)other);
        throw_if(res < 0 && PyErr_Occurred());
        return res;
    }
    int compare_ascii(const char *other) const requires std::same_as<T,_str>
    {
        return PyUnicode_CompareWithASCIIString((PyObject*)_ptr(), other);
    }
    const char *utf8() const requires std::same_as<T,_str>
    {
        return PyUnicode_AsUTF8((PyObject*)_ptr());
    }
    auto utf8_view() const requires std::same_as<T,_str>
    {
        Py_ssize_t size;
        auto data = PyUnicode_AsUTF8AndSize((PyObject*)_ptr(), &size);
        return std::string_view(data, size);
    }
    auto concat(auto &&s2) const requires std::same_as<T,_str>
    {
        return str_ref(throw_if_not(PyUnicode_Concat((PyObject*)_ptr(), (PyObject*)s2)));
    }
    auto split(auto &&sep, Py_ssize_t maxsplit) const requires std::same_as<T,_str>
    {
        return list_ref(throw_if_not(PyUnicode_Split((PyObject*)_ptr(),
                                                     (PyObject*)sep, maxsplit)));
    }
    auto join(auto &&items) const requires std::same_as<T,_str>
    {
        return str_ref(throw_if_not(PyUnicode_Join((PyObject*)_ptr(), (PyObject*)items)));
    }
    auto substr(Py_ssize_t start, Py_ssize_t end) const requires std::same_as<T,_str>
    {
        return str_ref(throw_if_not(PyUnicode_Substring((PyObject*)_ptr(),
                                                        start, end)));
    }

    template<typename T2>
    void add_objref(const char *name, T2 &&value) const requires std::same_as<T,_mod>
    {
#if PY_VERSION_HEX >= 0x030a0000
        throw_if(PyModule_AddObjectRef((PyObject*)_ptr(), name, (PyObject*)value) < 0);
#else
        py::ref v(newref(std::forward<T2>(value)));
        throw_if(PyModule_AddObject((PyObject*)_ptr(), name, v.get()) < 0);
        v.rel();
#endif
    }
    template<typename T2>
    void add_type(T2 &&value) const requires std::same_as<T,_mod>
    {
        throw_if(PyModule_AddType((PyObject*)_ptr(), (PyTypeObject*)value) < 0);
    }

private:
    auto *_ptr() const
    {
        return static_cast<const H<T>*>(this)->get();
    }
    template<template<typename> class H2, typename T2> friend struct common;
    template<typename T2> friend struct ptr;
    template<typename T2> friend struct _ref;
};

template<typename T>
struct ptr : common<ptr,T> {
    constexpr ptr() = default;
    constexpr ptr(auto *ptr) : m_ptr{(T*)ptr}
    {
    }
    template<template<typename> class H, typename T2>
    constexpr ptr(const common<H,T2> &h) : m_ptr{(T*)h._ptr()}
    {
    }
    ptr &operator=(auto *p) noexcept
    {
        m_ptr = (T*)p;
        return *this;
    }
    ptr &operator=(std::nullptr_t) noexcept
    {
        m_ptr = (T*)nullptr;
        return *this;
    }
    template<template<typename> class H, typename T2>
    ptr &operator=(const common<H,T2> &h) noexcept
    {
        m_ptr = (T*)h._ptr();
        return *this;
    }
    template<typename T2> ptr &operator=(ref<T2>&&) noexcept = delete;
    using common<ptr,T>::get;
    template<typename T2=T>
    constexpr auto *get() const
    {
        return (py_ptr_type<T2>*)m_ptr;
    }
    template<typename T2> constexpr operator T2*() const
    {
        return (T2*)m_ptr;
    }

private:
    T *m_ptr{nullptr};
    template<typename T2> friend struct ptr;
    template<typename T2> friend struct _ref;
};

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
        XDECREF(m_ptr);
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
    _ref &operator=(std::nullptr_t) noexcept
    {
        CLEAR(*this);
        return *this;
    }
    using common<_ref,T>::get;
    template<typename T2=T> constexpr auto *get() const
    {
        check_refcnt(m_ptr);
        return (py_ptr_type<T2>*)m_ptr;
    }
    template<typename T2>
    explicit constexpr operator T2*() const
    {
        check_refcnt(m_ptr);
        return (T2*)m_ptr;
    }
    void swap(_ref &other) noexcept
    {
        check_refcnt(m_ptr);
        check_refcnt(other.m_ptr);
        std::swap(m_ptr, other.m_ptr);
    }
    void take(auto *p) noexcept
    {
        check_refcnt(p);
        auto ptr = m_ptr;
        m_ptr = (T*)p;
        XDECREF(ptr);
    }
    template<typename T2>
    void take(ref<T2> &&h) noexcept
    {
        auto ptr = m_ptr;
        m_ptr = (T*)h.rel();
        XDECREF(ptr);
        check_refcnt(m_ptr);
    }
    template<typename T2>
    void assign(T2 &&p) noexcept
    {
        take(newref(std::forward<T2>(p)));
    }
    template<typename T2=T> auto *rel()
    {
        auto p = m_ptr;
        m_ptr = nullptr;
        check_refcnt(p);
        return (py_ptr_type<T2>*)p;
    }
    using common<_ref,T>::resize;
    void resize(Py_ssize_t sz) requires std::same_as<T,_bytes>
    {
        throw_if(_PyBytes_Resize((PyObject**)&m_ptr, sz));
    }

private:
    T *m_ptr{nullptr};
    template<typename T2> friend struct ptr;
    template<typename T2> friend struct _ref;
    template<typename T2> friend void CLEAR(ref<T2> &r);
};

template<typename T>
static inline void CLEAR(ref<T> &r)
{
    CLEAR(r.m_ptr);
}

template<template<typename> class H, typename T>
template<typename T2, typename T3>
inline auto common<H,T>::__ref(T3 *v)
{
    return py::ref<std::conditional_t<std::is_void_v<T2>,T3,T2>>(v);
}
template<template<typename> class H, typename T>
template<typename T2, typename T3>
inline auto common<H,T>::__ptr(T3 *v)
{
    return py::ptr<std::conditional_t<std::is_void_v<T2>,T3,T2>>(v);
}

template<bool exact=false, template<typename> class H, typename T, typename T3>
bool isa(const common<H,T> &self, T3 &&type)
{
    return self.template isa<exact>(std::forward<T3>(type));
}
template<bool exact=false, typename T3>
auto isa(auto *_self, T3 &&type)
{
    return isa<exact>(ptr(_self), std::forward<T3>(type));
}
template<typename T2, bool exact=false, template<typename> class H, typename T>
bool isa(const common<H,T> &self)
{
    return self.template isa<T2,exact>();
}
template<typename T2, bool exact=false>
bool isa(auto *_self)
{
    return isa<T2,exact>(ptr(_self));
}
template<typename T2, typename T> bool typeis(T &&self, T2 &&type)
{
    return isa<true>(std::forward<T>(self), std::forward<T2>(type));
}
template<typename T2, typename T> bool typeis(T &&self)
{
    return isa<T2,true>(std::forward<T>(self));
}

template<typename T2, bool exact=false, typename T, typename T3>
auto cast(ref<T> &&self, T3 &&type)
{
    if (isa<exact>(self, std::forward<T3>(type)))
        return ref<py_tag_type<T2>>(std::move(self));
    return ref<py_tag_type<T2>>();
}
template<typename T2, bool exact=false, template<typename> class H, typename T, typename T3>
auto cast(const common<H,T> &self, T3 &&type)
{
    if (isa<exact>(self, std::forward<T3>(type)))
        return self.template ptr<T2>();
    return ptr<py_tag_type<T2>>();
}
template<typename T2, bool exact=false, typename T3>
auto cast(auto *_self, T3 &&type)
{
    return cast<T2,exact>(ptr(_self), std::forward<T3>(type));
}

template<typename T2, bool exact=false, typename T>
auto cast(ref<T> &&self)
{
    if (isa<T2,exact>(self))
        return ref<py_tag_type<T2>>(std::move(self));
    return ref<py_tag_type<T2>>();
}
template<typename T2, bool exact=false, template<typename> class H, typename T>
auto cast(const common<H,T> &self)
{
    if (isa<T2,exact>(self))
        return self.template ptr<T2>();
    return ptr<py_tag_type<T2>>();
}
template<typename T2, bool exact=false>
auto cast(auto *_self)
{
    return cast<T2,exact>(ptr(_self));
}

template<typename T, typename T2, typename T3>
auto exact_cast(T2 &&self, T3 &&type)
{
    return cast<T,true>(std::forward<T2>(self), std::forward<T3>(type));
}
template<typename T, typename T2>
auto exact_cast(T2 &&self)
{
    return cast<T,true>(std::forward<T2>(self));
}

template<typename T, bool exact=false, typename T2, typename T3>
auto arg_cast(T2 &&self, T3 &&type, const char *name)
{
    if (auto res = cast<T,exact>(std::forward<T2>(self), std::forward<T3>(type)))
        return res;
    py_throw_format(PyExc_TypeError, "Unexpected type '%S' for %s",
                    Py_TYPE((PyObject*)self), name);
}
template<typename T, bool exact=false, typename T2>
auto arg_cast(T2 &&self, const char *name)
{
    if (auto res = cast<T,exact>(std::forward<T2>(self)))
        return res;
    py_throw_format(PyExc_TypeError, "Unexpected type '%S' for %s",
                    Py_TYPE((PyObject*)self), name);
}

template<typename T>
static inline void assign(auto *&field, T &&v)
{
    ref((PyObject*)std::exchange(field, newref(std::forward<T>(v))));
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

static inline ref<> new_none()
{
    return ref(immref(Py_None));
}

static inline ref<> new_bool(bool v)
{
    return ref(immref(v ? Py_True : Py_False));
}

static inline ref<> new_true()
{
    return new_bool(true);
}

static inline ref<> new_false()
{
    return new_bool(false);
}

static inline ref<> new_not_implemented()
{
    return ref(immref(Py_NotImplemented));
}

static inline bool is_slice_none(ptr<> key)
{
    if (!PySlice_Check(key))
        return false;
    auto slice = (PySliceObject*)key;
    return slice->start == Py_None && slice->stop == Py_None && slice->step == Py_None;
}

static inline auto new_dict()
{
    return dict_ref::checked(PyDict_New());
}
ref<> dict_deepcopy(ptr<> d);

template<typename T=PyObject*>
static inline auto new_set(T &&h=nullptr)
{
    return set_ref::checked(PySet_New((PyObject*)h));
}

static inline auto new_list(Py_ssize_t n)
{
    return list_ref::checked(PyList_New(n));
}
template<typename T, typename... Args>
static inline auto new_list(T &&first, Args&&... args)
    requires (!std::integral<std::remove_cvref_t<T>>)
{
    Py_ssize_t n = sizeof...(Args) + 1;
    auto res = new_list(n);
    PyObject *objs[] = { newref(std::forward<T>(first)),
        newref(std::forward<Args>(args))... };
    for (int i = 0; i < n; i++)
        res.SET(i, ref(objs[i]));
    return res;
}
static inline auto new_nlist(Py_ssize_t n, auto &&cb)
{
    auto res = new_list(n);
    for (int i = 0; i < n; i++)
        res.SET(i, cb(i));
    return res;
}

extern tuple empty_tuple;
static inline auto new_tuple(Py_ssize_t n)
{
    return tuple_ref::checked(PyTuple_New(n));
}
static inline auto new_tuple()
{
    return empty_tuple.immref();
}
template<typename T, typename... Args>
static inline auto new_tuple(T &&first, Args&&... args)
    requires (!std::integral<std::remove_cvref_t<T>>)
{
    Py_ssize_t n = sizeof...(Args) + 1;
    auto res = new_tuple(n);
    PyObject *objs[] = { newref(std::forward<T>(first)),
        newref(std::forward<Args>(args))... };
    for (int i = 0; i < n; i++)
        res.SET(i, ref(objs[i]));
    return res;
}
static inline auto new_ntuple(Py_ssize_t n, auto &&cb)
{
    auto res = new_tuple(n);
    for (int i = 0; i < n; i++)
        res.SET(i, cb(i));
    return res;
}

extern bytes empty_bytes;
static inline auto new_bytes(const char *data, Py_ssize_t len)
{
    return bytes_ref::checked(PyBytes_FromStringAndSize(data, len));
}
static inline auto new_bytes()
{
    return empty_bytes.immref();
}

static inline auto new_bytearray(const char *data, Py_ssize_t len)
{
    return bytearray_ref::checked(PyByteArray_FromStringAndSize(data, len));
}
static inline auto new_bytearray()
{
    return new_bytearray(nullptr, 0);
}

extern float_ float_m1;
extern float_ float_m0_5;
extern float_ float_0;
extern float_ float_0_5;
extern float_ float_1;
static inline auto new_float(double v)
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
    return float_ref::checked(PyFloat_FromDouble(v));
}

static constexpr int _int_cache_max = 4096;
extern const std::array<int_,_int_cache_max * 2> _int_cache;

template<std::integral T>
static constexpr bool _int_in_cache(T v)
{
    return v < _int_cache_max && (!std::is_signed_v<T> || v >= -_int_cache_max);
}

template<std::integral auto v>
static consteval void assert_int_cache()
{
    static_assert(_int_in_cache(v));
}

static inline auto int_cached(int v)
{
    assert(_int_in_cache(v));
    return _int_cache[v + _int_cache_max];
}

template<std::integral T>
static inline auto new_int(T v)
{
    if (_int_in_cache(v))
        return int_cached(v).ref();
    if constexpr (sizeof(v) > sizeof(long)) {
        return int_ref::checked(std::is_signed_v<T> ? PyLong_FromLongLong(v) :
                                PyLong_FromUnsignedLongLong(v));
    }
    else {
        return int_ref::checked(std::is_signed_v<T> ? PyLong_FromLong(v) :
                                PyLong_FromUnsignedLong(v));
    }
}

static inline auto new_str(const char *str)
{
    return str_ref::checked(PyUnicode_FromString(str));
}
static inline auto new_str(const char *str, Py_ssize_t len)
{
    return str_ref::checked(PyUnicode_FromStringAndSize(str, len));
}
static inline auto new_str(const std::string &str)
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
    template<size_t N, typename T>
    void write_cxx(T &&v)
    {
        std::array<char,N> str_buff;
        auto ptr = to_chars(str_buff, std::forward<T>(v));
        write_ascii(str_buff.data(), ptr - str_buff.data());
    }
    template<std::integral T>
    void write_hex(T v, bool showbase=false, int min_len=0, char fill='0')
    {
        std::array<char,sizeof(T) * 8> str_buff;
        if (showbase)
            write_ascii("0x");
        auto ptr = to_chars(str_buff, std::make_unsigned_t<T>(v), 16);
        auto len = ptr - str_buff.data();
        if (len < min_len)
            write_rep_ascii(min_len - len, &fill, 1);
        write_ascii(str_buff.data(), len);
    }
    std::pair<int,void*> reserve_buffer(int kind, ssize_t len);
    str_ref getvalue();
    stringio &operator<<(const char *s)
    {
        write_ascii(s);
        return *this;
    }
    stringio &operator<<(bool b)
    {
        write_ascii(b ? "true" : "false");
        return *this;
    }
    template<typename T>
    stringio &operator<<(T v) requires (std::integral<T> || std::floating_point<T>)
    {
        write_cxx<64>(v);
        return *this;
    }

private:
    void write_kind(const void *data, int kind, ssize_t len);
    void check_size(size_t sz, int kind);

    std::unique_ptr<char,CDeleter> m_buff;
    size_t m_size{0};
    size_t m_pos{0};
    int m_kind{PyUnicode_1BYTE_KIND};
};

struct bytesio {
    bytesio &operator=(const bytesio&) = delete;
    void write(const void *data, ssize_t len)
    {
        memcpy(reserve_buffer(len), data, len);
    }
    void *reserve_buffer(ssize_t len)
    {
        auto oldsz = m_buff.size();
        m_buff.resize(oldsz + len);
        return &m_buff.data()[oldsz];
    }
    bytes_ref &getvalue()
    {
        return m_buff;
    }

private:
    bytes_ref m_buff{new_bytes()};
};

static inline auto import_module(const char *str)
{
    return mod_ref::checked(PyImport_ImportModule(str));
}

static inline auto try_import_module(const char *str)
{
    auto mod = PyImport_ImportModule(str);
    if (!mod)
        PyErr_Clear();
    return mod_ref(mod);
}

static inline auto new_module(PyModuleDef *def)
{
    return mod_ref::checked(PyModule_Create(def));
}

template<str_literal modname,str_literal... names>
static inline auto imp()
{
    constexpr auto len = sizeof...(names);
    if constexpr (len == 0) {
        static mod m = import_module(modname).rel();
        return m;
    }
    else {
        static ptr res = [] <size_t... I> (std::index_sequence<I...>) {
            constexpr std::tuple name_tuple(names...);
            return imp<modname,std::get<I>(name_tuple)...>().attr((names,...)).rel();
        } (std::make_index_sequence<len - 1>());
        return res;
    }
}

#define PY_MODINIT(name, moddef)                                        \
    static inline void __PyInit_##name(brassboard_seq::py::mod);        \
    PyMODINIT_FUNC PyInit_##name(void) { return cxx_catch([] {          \
        brassboard_seq::init();                                         \
        auto m = brassboard_seq::py::new_module(&moddef);               \
        __PyInit_##name(m);                                             \
        return m;                                                       \
    }); }                                                               \
    static inline void __PyInit_##name(brassboard_seq::py::mod m)

template<typename T1=PyObject*,typename T2=PyObject*>
static inline auto new_cfunc(PyMethodDef *ml, T1 &&self=nullptr, T2 &&mod=nullptr)
{
    return ref<>::checked(PyCFunction_NewEx(ml, (PyObject*)self, (PyObject*)mod));
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

template<typename It, typename T, typename... Args>
struct _iter {
    template<typename... Args2>
    explicit _iter(T &&obj, Args2&&... args2)
        : obj(std::forward<T>(obj)), args(std::forward<Args2>(args2)...)
    {}
    auto begin()
    {
        return std::apply([] <typename... Args3> (Args3&&... args3) {
                return It(std::forward<Args3>(args3)...);
            }, arg_tuple());
    };
    auto end()
    {
        return std::apply([] <typename... Args3> (Args3&&... args3) {
                return It::end(std::forward<Args3>(args3)...);
            }, arg_tuple());
    }
private:
    T obj;
    std::tuple<Args...> args;
    auto arg_tuple()
    {
        return std::tuple_cat(std::tuple((PyObject*)obj), args);
    }
};

template<typename Value>
struct _generic_iterator {
    _generic_iterator(PyObject *it, uintptr_t key)
        : it(it), item(PyIter_Next(it))
    {
    }
    _generic_iterator &operator++()
    {
        item.take(PyIter_Next(it));
        throw_if(!item && PyErr_Occurred(), key);
        return *this;
    }
    ptr<py_tag_type<Value>> operator*()
    {
        return item.ptr();
    }
    bool operator==(std::nullptr_t) { return !item; }
    static std::nullptr_t end(auto&&...) { return nullptr; };

private:
    PyObject *it;
    ref<> item;
    uintptr_t key;
};

template<template<typename> class H, typename T>
template<typename Value>
inline auto common<H,T>::generic_iter(uintptr_t key) const
{
    auto it = throw_if_not(PyObject_GetIter((PyObject*)_ptr()), key);
    return _iter<_generic_iterator<Value>,py::ref<>,uintptr_t>(py::ref(it), key);
}

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
    std::pair<ptr<py_tag_type<Key>>,ptr<py_tag_type<Value>>> operator*()
    {
        return { ptr((py_tag_type<Key>*)key), ptr((py_tag_type<Value>*)value) };
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
    return _iter<_dict_iterator<Value,Key>,std::remove_cv_t<T>>(std::forward<T>(h));
}

template<typename Value>
struct _list_iterator {
    _list_iterator(PyObject *list) : list(list) {}
    _list_iterator &operator++()
    {
        ++pos;
        return *this;
    }
    std::pair<Py_ssize_t,ptr<py_tag_type<Value>>> operator*()
    {
        return { pos, ptr((py_tag_type<Value>*)PyList_GET_ITEM(list, pos)) };
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
    return _iter<_list_iterator<Value>,std::remove_cv_t<T>>(std::forward<T>(h));
}

template<typename Value>
struct _tuple_iterator {
    _tuple_iterator(PyObject *tuple) : tuple(tuple) {}
    _tuple_iterator &operator++()
    {
        ++pos;
        return *this;
    }
    std::pair<Py_ssize_t,ptr<py_tag_type<Value>>> operator*()
    {
        return { pos, ptr((py_tag_type<Value>*)PyTuple_GET_ITEM(tuple, pos)) };
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
    return _iter<_tuple_iterator<Value>,std::remove_cv_t<T>>(std::forward<T>(h));
}

template<template<typename> class H, typename T>
template<typename T2>
auto common<H,T>::append(T2 &&v) const requires std::same_as<T,_tuple>
{
    Py_ssize_t nele = size();
    auto res = new_tuple(nele + 1);
    for (auto p = _ptr(); auto [i, v]: tuple_iter(p))
        res.SET(i, v);
    res.SET(nele, std::forward<T2>(v));
    return res;
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
    return _iter<_str_iterator,std::remove_cv_t<T>>(std::forward<T>(h));
}

static inline bool isinstance_nontrivial(ptr<> obj, ptr<> ty)
{
    auto objt = obj.type();
    // Assume objt != ty and ty != object, and skip the first and last element in mro.
    // Also assume fully initialized type `ty`
    tuple mro = objt->tp_mro;
    for (Py_ssize_t i = 1, n = mro.size() - 1; i < n; i++) {
        if (mro.get(i) == ty) {
            return true;
        }
    }
    return false;
}

template<typename PyBase>
struct VBase : PyObject {
    template<typename T>
    struct Base : PyBase {
        const auto *data() const { return _data((const T*)this); }
        auto *data() { return _data((T*)this); }
        ~Base()
        {
            call_destructor(data());
        }
        template<typename... Args>
        static ref<T> alloc(Args&&... args)
        {
            auto self = generic_alloc<T>();
            call_constructor(self->data(), std::forward<Args>(args)...);
            return self;
        }
    };
    const auto *data() const { return _data((const PyBase*)this); }
    auto *data() { return _data((PyBase*)this); }
private:
    template<typename T>
    static auto *_data(T *p)
    {
        if constexpr (std::is_const_v<std::remove_reference_t<T>>) {
            return (const typename T::Data*)(((const char*)p) + sizeof(PyBase));
        }
        else {
            return (typename T::Data*)(((char*)p) + sizeof(PyBase));
        }
    }
};

[[noreturn]] void num_arg_error(const char *func_name, ssize_t nfound,
                                ssize_t nmin, ssize_t nmax);
[[noreturn]] void unexpected_kwarg_error(const char *func_name, str name);
static __attribute__((always_inline)) inline void
check_num_arg(const char *func_name, ssize_t nfound, ssize_t nmin, ssize_t nmax=-1)
{
    if ((nfound <= nmax || nmax < 0) && nfound >= nmin)
        return;
    num_arg_error(func_name, nfound, nmin, nmax);
}

static __attribute__((always_inline)) inline void
check_no_kwnames(const char *name, tuple kwnames)
{
    if (kwnames && kwnames.size()) {
        unexpected_kwarg_error(name, kwnames.get(0));
    }
}

static inline void check_required_pos_arg(ptr<> arg, const char *func,
                                          const char *name)
{
    if (!arg) {
        py_throw_format(PyExc_TypeError,
                        "%s missing 1 required positional argument: '%s'", func, name);
    }
}

static inline void check_non_empty_string(ptr<> arg, const char *name)
{
    if (auto s = cast<str>(arg); s && s.size())
        return;
    py_throw_format(PyExc_TypeError, "%s must be a non-empty string", name);
}

template<str_literal... argnames>
static inline auto parse_pos_or_kw_args(const char *fname, PyObject *const *args,
                                        Py_ssize_t nargs, tuple kwnames)
{
    std::array<ptr<>,sizeof...(argnames)> res;
    const char *argnames_ary[] = { argnames.value... };
    for (Py_ssize_t i = 0; i < nargs; i++)
        res[i] = args[i];
    if (kwnames) {
        auto kwargs = args + nargs;
        for (auto [i, kwname]: tuple_iter<str>(kwnames)) {
            bool found = false;
            for (size_t j = 0; j < sizeof...(argnames); j++) {
                if (kwname.compare_ascii(argnames_ary[j]) == 0) {
                    if (res[j])
                        py_throw_format(PyExc_TypeError,
                                        "%s got multiple values for argument '%s'",
                                        fname, argnames_ary[j]);
                    res[j] = kwargs[i];
                    found = true;
                    break;
                }
            }
            if (!found) {
                unexpected_kwarg_error(fname, kwname);
            }
        }
    }
    return res;
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
        py::ref<> get_traceback(PyObject *next);
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

    py::ref<> get_backtrace(uintptr_t key);
    ~BacktraceTracker()
    {
        // Do the freeing here instead of in the destructor of the FrameInfo object
        // so that we don't need to worry about the FrameInfo object being
        // copied/moved around when we add the frames.
        for (auto &[key, trace]: traces) {
            for (auto &frame: trace) {
                py::DECREF(frame.code);
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
uintptr_t event_time_key(py::ptr<> event_time)
{
    return (uintptr_t)(void*)event_time;
}
static constexpr __attribute__((always_inline,pure))
uintptr_t action_key(int aid)
{
    return (uintptr_t)(aid << 2) | 1;
}
static constexpr __attribute__((always_inline,pure))
uintptr_t assert_key(int aid)
{
    return (uintptr_t)(aid << 2) | 2;
}

void bb_reraise(uintptr_t key);

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

template<typename T=PyObject*>
auto cxx_catch(auto &&cb)
{
    using raw_ret_type = std::remove_cvref_t<decltype(cb())>;
    constexpr bool is_void = std::is_void_v<raw_ret_type>;
    constexpr bool is_ptr = std::is_pointer_v<raw_ret_type>;
    constexpr bool is_handle = py::is_handle_v<raw_ret_type>;
    constexpr bool is_ref = is_handle && requires { cb().rel(); };
    try {
        if constexpr (is_ref) {
            static_assert(std::is_pointer_v<T>);
            return (T)cb().rel();
        }
        else if constexpr (is_handle) {
            static_assert(std::is_pointer_v<T>);
            return (T)cb().ref().rel();
        }
        else if constexpr (!is_void) {
            static_assert(std::is_pointer_v<T> == is_ptr);
            return (T)cb();
        }
        else {
            cb();
            if constexpr (std::is_pointer_v<T>) {
                return (T)py::immref(Py_None);
            }
            else {
                static_assert(std::is_integral_v<T>);
                return (T)0;
            }
        }
    }
    catch (...) {
        handle_cxx_exception();
        if constexpr (std::is_pointer_v<T>) {
            return (T)nullptr;
        }
        else {
            static_assert(std::is_integral_v<T>);
            return (T)-1;
        }
    }
}

namespace py {

template<auto F>
static inline PyObject *unifunc(PyObject *v1)
{
    return cxx_catch([&] { return F(v1); });
}

template<auto F>
static inline PyObject *binfunc(PyObject *v1, PyObject *v2)
{
    return cxx_catch([&] { return F(v1, v2); });
}

template<auto F>
static inline PyObject *trifunc(PyObject *v1, PyObject *v2, PyObject *v3)
{
    return cxx_catch([&] { return F(v1, v2, v3); });
}

template<auto F, typename I=int>
static inline I iunifunc(PyObject *v1)
{
    return cxx_catch<I>([&] { return F(v1); });
}

template<auto F, typename I=int>
static inline I ibinfunc(PyObject *v1, PyObject *v2)
{
    return cxx_catch<I>([&] { return F(v1, v2); });
}

template<auto F, typename I=int>
static inline I itrifunc(PyObject *v1, PyObject *v2, PyObject *v3)
{
    return cxx_catch<I>([&] { return F(v1, v2, v3); });
}

template<str_literal name, auto F, int flags, str_literal doc>
struct _method_def {
    constexpr operator PyMethodDef() const
    {
        return {name, (PyCFunction)(uintptr_t)F, flags, doc};
    }
};

template<auto F> static constexpr auto cfunc = binfunc<F>;
template<str_literal name, auto F, str_literal doc="",int flags=0>
static constexpr auto meth_o = _method_def<name,cfunc<F>,METH_O|flags,doc>{};

template<auto F>
static inline PyObject *cfunc_noargs(PyObject *self, PyObject *arg)
{
    assert(!arg);
    return cxx_catch([&] { return F(self); });
}
template<str_literal name, auto F, str_literal doc="",int flags=0>
static constexpr auto meth_noargs = _method_def<name,cfunc_noargs<F>,METH_NOARGS|flags,doc>{};

template<auto F>
static inline PyObject *cfunc_fast(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    return cxx_catch([&] { return F(self, args, nargs); });
}
template<str_literal name, auto F, str_literal doc="",int flags=0>
static constexpr auto meth_fast = _method_def<name,cfunc_fast<F>,METH_FASTCALL|flags,doc>{};

template<auto F>
static inline PyObject *cfunc_fastkw(PyObject *self, PyObject *const *args,
                                     Py_ssize_t nargs, PyObject *kwnames)
{
    return cxx_catch([&] { return F(self, args, nargs, kwnames); });
}
template<str_literal name, auto F, str_literal doc="",int flags=0>
static constexpr auto meth_fastkw = _method_def<name,cfunc_fastkw<F>,
                                                METH_FASTCALL|METH_KEYWORDS|flags,doc>{};

template<_method_def... defs> static inline PyMethodDef meth_table[] = { defs..., {} };

template<str_literal name, int type, auto ptr, int flags, str_literal doc>
struct _mem_def;
template<str_literal name, int type, typename T, typename Fld, Fld T::*ptr,
         int flags, str_literal doc>
struct _mem_def<name,type,ptr,flags,doc>
{
    operator PyMemberDef() const
    {
        T v;
        return {name, type, ((char*)&(v.*ptr)) - (char*)&v, flags, doc};
    }
};
template<str_literal name, int type, auto ptr, int flags, str_literal doc="">
static constexpr _mem_def<name,type,ptr,flags,doc> mem_def;
template<_mem_def... defs> static inline PyMemberDef mem_table[] = { defs..., {} };

template<auto F>
static inline PyObject *getter_func(PyObject *self, void*)
{
    return cxx_catch([&] { return F(self); });
}

template<auto F>
static inline int setter_func(PyObject *self, PyObject *val, void*)
{
    return cxx_catch<int>([&] { return F(self, val); });
}

template<str_literal name, auto get, auto set, str_literal doc>
struct _getset_def {
    constexpr operator PyGetSetDef() const
    {
        if constexpr (std::is_null_pointer_v<decltype(set)>) {
            return {name, getter_func<get>, set, doc, nullptr};
        }
        else {
            return {name, getter_func<get>, setter_func<set>, doc, nullptr};
        }
    }
};
template<str_literal name, auto get, auto set=nullptr, str_literal doc="">
static constexpr _getset_def<name,get,set,doc> getset_def;
template<_getset_def... defs> static inline PyGetSetDef getset_table[] = { defs..., {} };

template<auto F>
static inline PyObject *vectorfunc(PyObject *self, PyObject *const *args,
                                   size_t nargsf, PyObject *kwnames)
{
    return cxx_catch([&] { return F(self, args, PyVectorcall_NARGS(nargsf), kwnames); });
}

template<auto F>
static inline PyObject *tp_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    return cxx_catch([&] { return F(type, args, kwargs); });
}

template<auto F>
static inline PyObject *tp_richcompare(PyObject *v1, PyObject *v2, int op)
{
    return cxx_catch([&] { return F(v1, v2, op); });
}

template<auto F>
static inline int sq_ass_item(PyObject *o, Py_ssize_t i, PyObject *v)
{
    return cxx_catch<int>([&] { return F(o, i, v); });
}

template<auto F>
static inline PyObject *sq_item(PyObject *o, Py_ssize_t i)
{
    return cxx_catch([&] { return F(o, i); });
}

template<bool gc, typename T>
static inline void tp_cxx_dealloc(PyObject *obj)
{
    if constexpr (gc)
        PyObject_GC_UnTrack(obj);
    auto t = Py_TYPE(obj);
    cxx_catch<int>([&] { call_destructor((T*)obj); });
    t->tp_free(obj);
}

struct tp_visitor {
    tp_visitor(visitproc visit, void *arg)
        : visit(visit),
          arg(arg)
    {}
    tp_visitor(const tp_visitor&) = delete;
    void operator()(auto &&obj)
    {
        if (res || !obj) [[unlikely]]
            return;
        res = visit((PyObject*)obj, arg);
    }

    int res{0};
private:
    const visitproc visit;
    void *const arg;
};

template<auto F>
static inline int tp_traverse(PyObject *self, visitproc visit, void *arg)
{
    tp_visitor visitor(visit, arg);
    F(self, visitor);
    return visitor.res;
}

template<typename Flds> struct _field_visit {};
template<typename T, auto ...flds> struct _field_visit<_field_pack<T,flds...>> {
    static inline void visit(py::ptr<T> self, tp_visitor &visitor)
    {
        (void)self; (void)visitor;
        (visitor(self.get()->*flds),...);
    }
};

template<typename T>
static constexpr auto field_pack_visit = _field_visit<T>::visit;
template<typename T, auto... fld>
static constexpr auto field_visit = field_pack_visit<field_pack<T,fld...>>;
template<typename T>
static constexpr auto tp_field_pack_traverse = tp_traverse<field_pack_visit<T>>;
template<typename T, auto... fld>
static constexpr auto tp_field_traverse = tp_field_pack_traverse<field_pack<T,fld...>>;

template<typename Flds> struct _field_clear {};
template<typename T, auto ...flds> struct _field_clear<_field_pack<T,flds...>> {
    static inline void clear(py::ptr<T> self)
    {
        (void)self;
        (CLEAR(self.get()->*flds),...);
    }
};

template<typename T>
static constexpr auto field_pack_clear = _field_clear<T>::clear;
template<typename T, auto... fld>
static constexpr auto field_clear = field_pack_clear<field_pack<T,fld...>>;
template<typename T>
static constexpr auto tp_field_pack_clear = iunifunc<field_pack_clear<T>>;
template<typename T, auto... fld>
static constexpr auto tp_field_clear = tp_field_pack_clear<field_pack<T,fld...>>;

} // py

template<typename T> static inline auto to_py(const T &v);

namespace py {

template<typename> struct converter {
    static inline ptr<> py(bool b)
    {
        return b ? Py_True : Py_False;
    }
    static inline auto py(std::integral auto i)
    {
        return new_int(i);
    }
    static inline auto py(std::floating_point auto f)
    {
        return new_float(f);
    }
    template<typename T1, typename T2>
    static inline auto py(const std::pair<T1,T2> &p)
    {
        return new_tuple(to_py(p.first), to_py(p.second));
    }
    template<typename T>
    static inline auto py(const std::vector<T> &v)
    {
        return new_nlist(v.size(), [&] (int i) { return to_py(v[i]); });
    }
    template<typename K, typename V>
    static inline auto py(const std::map<K,V> &m)
    {
        auto res = new_dict();
        for (auto [k, v]: m)
            res.set(to_py(k), to_py(v));
        return res;
    }
};

template<typename T, size_t N> struct converter<T[N]> {
    static inline auto py(const auto *v)
    {
        return new_nlist(N, [&] (int i) { return to_py(v[i]); });
    }
};

} // py

template<typename T> static inline auto to_py(const T &v)
{
    return py::converter<std::remove_cvref_t<T>>::py(v);
}

template<str_literal lit>
static const py::str _py_string_cache(throw_if_not(PyUnicode_InternFromString(lit.value)));
template<str_literal lit>
static inline auto operator ""_py()
{
    return _py_string_cache<lit>;
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

struct IsFirst {
    bool first{true};
    bool get()
    {
        return std::exchange(first, false);
    }
};

template<typename T, size_t N>
class PermAllocator {
public:
    struct iterator {
        T **pages;
        T **const pages_last;
        T *page;
        size_t ele_idx{0};
        size_t const last_page_cnt;

        iterator(PermAllocator &allocator)
            : pages(allocator.pages.data()),
              pages_last(pages + allocator.pages.size() - 1),
              page(allocator.pages.size() ? assume(pages[0]) : nullptr),
              last_page_cnt(N - allocator.space_left)
        {}
        iterator &operator++()
        {
            ele_idx++;
            assume(page);
            if (pages == pages_last) {
                if (ele_idx == last_page_cnt) {
                    page = nullptr;
                }
            }
            else if (ele_idx == N) {
                pages++;
                page = assume(pages[0]);
                ele_idx = 0;
            }
            return *this;
        }
        T &operator*()
        {
            return page[ele_idx];
        }
        bool operator==(std::nullptr_t) const { return !page; }
    };
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

    auto begin()
    {
        return iterator(*this);
    }
    auto end()
    {
        return nullptr;
    }

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
        // Reverse order and use unsigned comparison
        return std::lexicographical_compare_three_way(
            bits.rbegin(), bits.rend(), other.bits.rbegin(), other.bits.rend(),
            [] (UELT v1, UELT v2) { return v1 <=> v2; });
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
    void print(py::stringio &io, bool showbase=false) const
    {
        if (showbase)
            io << "0x";
        for (auto v: std::ranges::views::reverse(bits)) {
            io.write_hex(v, false, elbits / 4);
        }
    }
    auto to_pybytes() const
    {
        return py::new_bytes((const char*)&bits[0], sizeof(bits));
    }
    auto to_pylong() const
    {
#if PY_VERSION_HEX >= 0x030d0000
        return py::ref<>::checked(PyLong_FromUnsignedNativeBytes(&bits[0],
                                                                 sizeof(bits), 1));
#else
        return py::ref<>::checked(_PyLong_FromByteArray((const unsigned char*)&bits[0],
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
    template<std::integral ELT2,unsigned N2> friend struct Bits;
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

struct cubic_spline {
    double order0;
    double order1;
    double order2;
    double order3;
    constexpr bool operator==(const cubic_spline &other) const
    {
        return (order0 == other.order0) && (order1 == other.order1) &&
            (order2 == other.order2) && (order3 == other.order3);
    }
    constexpr std::array<double,4> to_array() const
    {
        return { order0, order1, order2, order3 };
    }
    constexpr OPT_FAST_MATH __attribute__((always_inline))
    double eval(double t) const
    {
        return order0 + (order1 + (order2 + order3 * t) * t) * t;
    }
    constexpr OPT_FAST_MATH __attribute__((always_inline))
    cubic_spline resample(double t1, double t2) const
    {
        double dt = t2 - t1;
        double dt2 = dt * dt;
        double dt3 = dt2 * dt;
        double o3_3 = 3 * order3;
        return {
            order0 + (order1 + (order2 + order3 * t1) * t1) * t1,
            dt * (order1 + (2 * order2 + o3_3 * t1) * t1),
            dt2 * (order2 + o3_3 * t1), dt3 * order3,
        };
    }
    constexpr cubic_spline resample_cycle(int64_t start, int64_t end,
                                          int64_t cycle1, int64_t cycle2) const
    {
        if (cycle1 == start && cycle2 == end)
            return *this;
        return resample(double(cycle1 - start) / double(end - start),
                        double(cycle2 - start) / double(end - start));
    }

    static constexpr __attribute__((always_inline)) cubic_spline from_static(double v0)
    {
        return { v0, 0, 0, 0 };
    }

    static constexpr OPT_FAST_MATH __attribute__((always_inline))
    cubic_spline from_values(double v0, double v1, double v2, double v3)
    {
        // v = o0 + o1 * t + o2 * t^2 + o3 * t^3

        // v0 = o0
        // v1 = o0 + o1 / 3 + o2 / 9 + o3 / 27
        // v2 = o0 + o1 * 2 / 3 + o2 * 4 / 9 + o3 * 8 / 27
        // v3 = o0 + o1 + o2 + o3

        // o0 = v0
        // o1 = -5.5 * v0 + 9 * v1 - 4.5 * v2 + v3
        // o2 = 9 * v0 - 22.5 * v1 + 18 * v2 - 4.5 * v3
        // o3 = -4.5 * v0 + 13.5 * v1 - 13.5 * v2 + 4.5 * v3

        return {
            v0,
            -5.5 * v0 + 9 * v1 - 4.5 * v2 + v3,
            9 * v0 - 22.5 * v1 + 18 * v2 - 4.5 * v3,
            -4.5 * v0 + 13.5 * v1 - 13.5 * v2 + 4.5 * v3,
        };
    }
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

void init();
extern PyMethodDef utils_methods[];

}

#endif
