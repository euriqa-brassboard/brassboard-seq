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

#include "utils.h"

#include <stdarg.h>
#include <stdlib.h>
#include <strings.h>

#include <algorithm>

namespace brassboard_seq {

__attribute__((visibility("protected")))
BBLogLevel bb_logging_level = [] {
    if (auto env = getenv("BB_LOG")) {
        if (strcasecmp(env, "debug") == 0) {
            return BB_LOG_DEBUG;
        }
        else if (strcasecmp(env, "info") == 0) {
            return BB_LOG_INFO;
        }
    }
    return BB_LOG_INFO;
}();

__attribute__((visibility("protected")))
PyMethodDef utils_methods[] = {
    py::meth_o<"set_log_level",[] (auto, py::ptr<> _level) {
        auto level = py::arg_cast<py::str>(_level, "level");
        if (level.compare_ascii("debug") == 0) {
            bb_logging_level = BB_LOG_DEBUG;
        }
        else if (level.compare_ascii("info") == 0 || level.compare_ascii("") == 0) {
            bb_logging_level = BB_LOG_INFO;
        }
        else {
            py_throw_format(PyExc_ValueError, "Invalid log level %U", level);
        }
    }>, {}};

__attribute__((visibility("protected")))
void format_double(std::ostream &io, double v)
{
    // Unlike `operator<<`, which uses a fixed precision (6 by default),
    // `std::to_chars` of floating point number (no precision specified)
    // is guaranteed to use the shortest accurate representation
    // of the number.
    // With C++23, we could use `std::print(io, "{}", order)` instead.
    // (Not using std::format since GCC 11.1 for artiq-7 nix environment
    //  doesn't have it)
    char buff[64];
    auto ptr = to_chars(buff, v);
    io.write(buff, ptr - buff);
}

namespace py {

#if PY_VERSION_HEX >= 0x030d0000
__attribute__((visibility("protected")))
tuple empty_tuple(Py_GetConstant(Py_CONSTANT_EMPTY_TUPLE));
__attribute__((visibility("protected")))
bytes empty_bytes(Py_GetConstant(Py_CONSTANT_EMPTY_BYTES));
#else
__attribute__((visibility("protected")))
tuple empty_tuple(new_tuple(0).rel());
__attribute__((visibility("protected")))
bytes empty_bytes(new_bytes(nullptr, 0).rel());
#endif

__attribute__((visibility("protected")))
float_ float_m1(throw_if_not(PyFloat_FromDouble(-1)));
__attribute__((visibility("protected")))
float_ float_m0_5(throw_if_not(PyFloat_FromDouble(-0.5)));
__attribute__((visibility("protected")))
float_ float_0(throw_if_not(PyFloat_FromDouble(0)));
__attribute__((visibility("protected")))
float_ float_0_5(throw_if_not(PyFloat_FromDouble(0.5)));
__attribute__((visibility("protected")))
float_ float_1(throw_if_not(PyFloat_FromDouble(1)));

const std::array<int_,_int_cache_max * 2> _int_cache = [] {
    std::array<int_,_int_cache_max * 2> res;
    for (int i = 0; i < _int_cache_max * 2; i++)
        res[i] = throw_if_not(PyLong_FromLong(i - _int_cache_max));
    return res;
} ();

static dict_ref _dict_deepcopy(dict d)
{
    auto res = new_dict();
    for (auto [key, value]: dict_iter(d)) {
        if (!value.isa<dict>()) {
            res.set(key, value);
            continue;
        }
        res.set(key, _dict_deepcopy(value));
    }
    return res;
}

__attribute__((visibility("protected")))
ref<> dict_deepcopy(ptr<> d)
{
    if (!d.isa<dict>())
        return d.ref();
    return _dict_deepcopy(d);
}

static inline void _copy_pystr_same_kind(void *tgt, const void *src, int kind, ssize_t len)
{
    memcpy(tgt, src, len * kind);
}

static inline void _copy_pystr_change_kind(void *tgt, int tgt_kind,
                                           const void *src, int src_kind, ssize_t len)
{
    for (ssize_t i = 0; i < len; i++) {
        PyUnicode_WRITE(tgt_kind, tgt, i, PyUnicode_READ(src_kind, src, i));
    }
}

static inline void _copy_pystr_buffer(void *tgt, int tgt_kind,
                                      const void *src, int src_kind, ssize_t len)
{
    if (tgt_kind == src_kind) [[likely]] {
        _copy_pystr_same_kind(tgt, src, tgt_kind, len);
    }
    else {
        _copy_pystr_change_kind(tgt, tgt_kind, src, src_kind, len);
    }
}

__attribute__((visibility("internal")))
inline void stringio::check_size(size_t sz, int kind)
{
    if (kind > m_kind) [[unlikely]] {
        auto new_buff = (char*)malloc(sz * kind);
        _copy_pystr_change_kind(new_buff, kind, m_buff.get(), m_kind, m_pos);
        m_kind = kind;
        m_size = sz;
        m_buff.reset(new_buff);
        return;
    }
    if (m_size >= sz)
        return;
    m_size = sz * 3 / 2;
    m_buff.reset((char*)realloc(m_buff.release(), m_size * m_kind));
}

__attribute__((visibility("internal")))
inline void stringio::write_kind(const void *data, int kind, ssize_t len)
{
    static_assert(PyUnicode_1BYTE_KIND == 1 &&
                  PyUnicode_2BYTE_KIND == 2 &&
                  PyUnicode_4BYTE_KIND == 4);
    check_size(m_pos + len, kind);
    _copy_pystr_buffer(m_buff.get() + m_pos * m_kind, m_kind, data, kind, len);
    m_pos += len;
}

__attribute__((visibility("protected")))
void stringio::write(str s)
{
    write_kind(PyUnicode_DATA(s.get()), PyUnicode_KIND(s.get()), s.size());
}

__attribute__((visibility("protected")))
void stringio::write_ascii(const char *s, ssize_t len)
{
    write_kind(s, PyUnicode_1BYTE_KIND, len);
}

__attribute__((visibility("protected")))
void stringio::write_rep_ascii(int nrep, const char *s, ssize_t len)
{
    static_assert(PyUnicode_1BYTE_KIND == 1);
    check_size(m_pos + len * nrep, PyUnicode_1BYTE_KIND);
    if (m_kind == PyUnicode_1BYTE_KIND) [[likely]] {
        for (int i = 0; i < nrep; i++) {
            _copy_pystr_same_kind(m_buff.get() + m_pos + len * i,
                                  s, PyUnicode_1BYTE_KIND, len);
        }
    }
    else {
        for (int i = 0; i < nrep; i++) {
            _copy_pystr_change_kind(m_buff.get() + (m_pos + len * i) * m_kind, m_kind,
                                    s, PyUnicode_1BYTE_KIND, len);
        }
    }
    m_pos += len * nrep;
}

__attribute__((visibility("protected")))
std::pair<int,void*> stringio::reserve_buffer(int kind, ssize_t len)
{
    check_size(m_pos + len, kind);
    auto ptr = m_buff.get() + m_pos * m_kind;
    m_pos += len;
    return {m_kind, ptr};
}

__attribute__((visibility("protected")))
str_ref stringio::getvalue()
{
    if (!m_pos)
        return ""_py.ref();
    return str_ref::checked(PyUnicode_FromKindAndData(m_kind, m_buff.get(), m_pos));
}

[[noreturn]] __attribute__((visibility("protected")))
void num_arg_error(const char *func_name, ssize_t nfound,
                   ssize_t nmin, ssize_t nmax)
{
    ssize_t nexpected;
    const char *more_or_less;
    if (nmin == nmax) {
        nexpected = nmin;
        more_or_less = "exactly";
    }
    else if (nfound < nmin) {
        nexpected = nmin;
        more_or_less = "at least";
    }
    else {
        nexpected = nmax;
        more_or_less = "at most";
    }
    py_throw_format(PyExc_TypeError,
                    "%.200s() takes %.8s %zd positional argument%.1s (%zd given)",
                    func_name, more_or_less, nexpected,
                    (nexpected == 1) ? "" : "s", nfound);
}

[[noreturn]] __attribute__((visibility("protected")))
void unexpected_kwarg_error(const char *func_name, py::str name)
{
    py_throw_format(PyExc_TypeError, "%s got an unexpected keyword argument '%U'",
                    func_name, name);
}

}

#if PY_VERSION_HEX < 0x030b00f0

static inline PyCodeObject *pyframe_getcode(PyFrameObject *frame)
{
    return (PyCodeObject*)py::xnewref((PyObject*)frame->f_code);
}
static inline int pyframe_getlasti(PyFrameObject *frame)
{
    return frame->f_lasti;
}
static inline PyFrameObject *pyframe_getback(PyFrameObject *frame)
{
    return (PyFrameObject*)py::xnewref((PyObject*)frame->f_back);
}

#else

static inline PyCodeObject *pyframe_getcode(PyFrameObject *frame)
{
    return PyFrame_GetCode(frame);
}
static inline int pyframe_getlasti(PyFrameObject *frame)
{
    return PyFrame_GetLasti(frame);
}
static inline PyFrameObject *pyframe_getback(PyFrameObject *frame)
{
    return PyFrame_GetBack(frame);
}

#endif

BacktraceTracker *BacktraceTracker::global_tracker;

static auto traceback_new = PyTraceBack_Type.tp_new;

BacktraceTracker::FrameInfo::FrameInfo(PyFrameObject *frame)
    : code(pyframe_getcode(frame)),
      lasti(pyframe_getlasti(frame)),
      lineno(PyFrame_GetLineNumber(frame))
{
}

py::ref<> BacktraceTracker::FrameInfo::get_traceback(PyObject *next)
{
    PyThreadState *tstate = PyThreadState_Get();
    auto globals = py::new_dict();
    auto args = py::new_tuple(
        py::ptr(next),
        py::ref<>::checked(PyFrame_New(tstate, code, globals.get(), nullptr)),
        py::new_int(lasti), py::new_int(lineno));
    return py::ref<>::checked(traceback_new(&PyTraceBack_Type, args.get(), nullptr));
}

void BacktraceTracker::_record(uintptr_t key)
{
    PyFrameObject *frame = PyEval_GetFrame();
    if (!frame)
        return;
    auto &trace = traces[key];
    // Borrowed frame reference, no need to free
    trace.push_back({frame});
    bool frame_need_free = false;
    for (int i = 1; i < max_frame; i++) {
        auto new_frame = pyframe_getback(frame);
        if (!new_frame)
            break;
        if (frame_need_free)
            py::DECREF(frame);
        trace.push_back({new_frame});
        frame = new_frame;
        frame_need_free = true;
    }
    if (frame_need_free) {
        py::DECREF(frame);
    }
}

py::ref<> BacktraceTracker::get_backtrace(uintptr_t key)
{
    assert(traceback_new);
    auto it = traces.find(key);
    if (it == traces.end())
        return py::ref();
    auto &trace = it->second;
    py::ref py_trace;
    for (auto &info: trace) {
        try {
            py_trace.take(info.get_traceback(py_trace ? py_trace.get() : Py_None));
        }
        catch (...) {
            // Skip a frame if we couldn't construct it.
            PyErr_Clear();
        }
    }
    return py_trace;
}

static py::ref<> combine_traceback(py::ref<> old_tb, py::ref<> tb)
{
    if (!old_tb)
        return tb;
    if (tb) {
        auto last_tb = old_tb.ptr<PyTracebackObject>();
        while (last_tb->tb_next)
            last_tb = last_tb->tb_next;
        last_tb->tb_next = (PyTracebackObject*)tb.rel();
    }
    return old_tb;
}

static inline auto get_global_backtrace(uintptr_t key)
{
    if (BacktraceTracker::global_tracker)
        return BacktraceTracker::global_tracker->get_backtrace(key);
    return py::ref();
}

[[noreturn]] __attribute__((visibility("protected")))
void throw0()
{
    throw 0;
}

__attribute__((visibility("protected")))
void bb_reraise(uintptr_t key)
{
    PyObject *exc, *type, *old_tb;
    PyErr_Fetch(&type, &exc, &old_tb);
    PyErr_Restore(type, exc, combine_traceback(py::ref(old_tb),
                                               get_global_backtrace(key)).rel());
}

[[noreturn]] __attribute__((visibility("protected")))
void bb_rethrow(uintptr_t key)
{
    bb_reraise(key);
    throw0();
}

[[noreturn]] __attribute__((visibility("protected")))
void _bb_throw_format(PyObject *exc, uintptr_t key,
                      const char *format, ...)
{
    // This is slightly less efficient but much simpler to implement.
    va_list vargs;
    va_start(vargs, format);
    PyErr_FormatV(exc, format, vargs);
    va_end(vargs);
    bb_rethrow(key);
}

[[noreturn]] __attribute__((visibility("protected")))
void _py_throw_format(PyObject *exc, const char *format, ...)
{
    // This is slightly less efficient but much simpler to implement.
    va_list vargs;
    va_start(vargs, format);
    PyErr_FormatV(exc, format, vargs);
    va_end(vargs);
    throw0();
}

__attribute__((visibility("protected")))
void handle_cxx_exception()
{
    if (PyErr_Occurred())
        return;
    try {
        throw;
    }
    catch (const std::bad_alloc& exn) {
        PyErr_SetString(PyExc_MemoryError, exn.what());
    }
    catch (const std::bad_cast& exn) {
        PyErr_SetString(PyExc_TypeError, exn.what());
    }
    catch (const std::bad_typeid& exn) {
        PyErr_SetString(PyExc_TypeError, exn.what());
    }
    catch (const std::domain_error& exn) {
        PyErr_SetString(PyExc_ValueError, exn.what());
    }
    catch (const std::invalid_argument& exn) {
        PyErr_SetString(PyExc_ValueError, exn.what());
    }
    catch (const std::ios_base::failure& exn) {
        PyErr_SetString(PyExc_IOError, exn.what());
    }
    catch (const std::out_of_range& exn) {
        PyErr_SetString(PyExc_IndexError, exn.what());
    }
    catch (const std::overflow_error& exn) {
        PyErr_SetString(PyExc_OverflowError, exn.what());
    }
    catch (const std::range_error& exn) {
        PyErr_SetString(PyExc_ArithmeticError, exn.what());
    }
    catch (const std::underflow_error& exn) {
        PyErr_SetString(PyExc_ArithmeticError, exn.what());
    }
    catch (const std::exception& exn) {
        PyErr_SetString(PyExc_RuntimeError, exn.what());
    }
    catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "Unknown exception");
    }
}

std::streamsize buff_streambuf::xsputn(const char *s, std::streamsize count)
{
    auto p = extend(count);
    memcpy(p, s, count);
    pbump((int)count);
    update_size();
    return count;
}

auto buff_streambuf::overflow(int_type ch) -> int_type
{
    // The document on this function is really confusing.
    // In terms of what it should do when `ch` is `eof` and
    // what to do about the current pointer
    //
    // [cppreference.com](https://en.cppreference.com/w/cpp/io/basic_streambuf/overflow)
    // says:
    // > Ensures that there is space at the put area for at least one character
    // which seems to imply that the reservation needs to be done even when `ch` is `eof`.
    //
    // [cplusplus.com](http://www.cplusplus.com/reference/streambuf/streambuf/overflow/)
    // says:
    // > put a character into the controlled output sequence
    // > without changing the current position.
    // Mentioning nothing about what needs to be done with `eof` as input and
    // seems to suggest that the current position should not be updated
    // after a character is added.
    //
    // From [a stackoverflow question](https://stackoverflow.com/questions/19849143/what-is-wrong-with-my-implementation-of-overflow)
    // and `libc++`s implementation it seems that `eof` input should just be ignored and
    // the current pointer should be updated after the character is written
    // so that's what we'll do....
    if (traits_type::eq_int_type(ch, traits_type::eof()))
        return traits_type::not_eof(ch);
    *extend(1) = (char)ch;
    pbump(1);
    update_size();
    return traits_type::not_eof(ch);
}

auto buff_streambuf::seekpos(pos_type pos, std::ios_base::openmode which) -> pos_type
{
    if (which != std::ios_base::out)
        return pos_type(-1);
    return _seekpos(pos);
}

__attribute__((visibility("internal")))
inline auto buff_streambuf::_seekpos(pos_type pos) -> pos_type
{
    if (pos < 0)
        return pos_type(-1);
    // Update before changing the pointer as well as after changing the pointer
    // so that we can catch seeking back.
    update_size();
    auto base = pbase();
    auto end = epptr();
    if (base + (off_type)pos > end) [[unlikely]] {
        extend(base + (off_type)pos - end);
        base = pbase();
    }
    pbump(int((off_type)pos - (pptr() - base)));
    update_size();
    return pos;
}

auto buff_streambuf::seekoff(off_type off, std::ios_base::seekdir dir,
                             std::ios_base::openmode which) -> pos_type
{
    if (which != std::ios_base::out)
        return pos_type(-1);
    pos_type pos;
    if (dir == std::ios_base::beg) {
        pos = off;
    }
    else if (dir == std::ios_base::cur) {
        pos = pptr() - pbase();
        if (off == 0)
            return pos;
        pos = pos + off;
    }
    else if (dir == std::ios_base::end) {
        if (off > 0)
            return pos_type(-1);
        pos = m_end + off;
    }
    else {
        return pos_type(-1);
    }
    return _seekpos(pos);
}

__attribute__((visibility("internal")))
inline int buff_streambuf::sync()
{
    update_size();
    return 0;
}

__attribute__((visibility("internal")))
inline void buff_streambuf::update_size()
{
    auto sz = pptr() - pbase();
    if (sz > m_end) {
        m_end = sz;
    }
}

pybytes_streambuf::pybytes_streambuf()
{
    setp(nullptr, nullptr);
}

pybytes_streambuf::~pybytes_streambuf()
{
}

py::bytes_ref pybytes_streambuf::get_buf()
{
    if (!m_buf)
        return py::new_bytes();
    auto sz = m_end;
    setp(nullptr, nullptr);
    m_buf.resize(sz);
    m_end = 0;
    return std::move(m_buf);
}

char *pybytes_streambuf::extend(size_t sz)
{
    auto oldbase = pbase();
    auto oldptr = pptr();
    auto oldsz = oldptr - oldbase;
    // overallocate.
    auto new_sz = (oldsz + sz) * 3 / 2;
    if (oldbase + new_sz <= epptr())
        return &m_buf.data()[oldsz];
    if (!m_buf) {
        m_buf.take(py::new_bytes(nullptr, new_sz));
    }
    else {
        m_buf.resize(new_sz);
    }
    auto buf = m_buf.data();
    setp(buf, &buf[new_sz]);
    pbump((int)oldsz);
    return &buf[oldsz];
}

pybytes_ostream::pybytes_ostream()
    : buff_ostream(&m_buf)
{
}

pybytes_ostream::~pybytes_ostream()
{
}

static void _get_suffix_array(std::span<int> SA, std::span<int> S, std::span<int> ws);
void get_suffix_array(std::span<int> SA, std::span<int> S, std::span<int> ws)
{
    int N = S.size();
    if (N <= 1) [[unlikely]] {
        if (N == 1) {
            assert(S[0] == 0);
            SA[0] = 0;
        }
        return;
    }
    _get_suffix_array(SA, S, ws);
}

/**
 * Classify characters (L/S/LMS), construct bucket
 *    a. Clear `bucket_cnt`
 *    b. Iterate `S` to count bucket size using `bucket_cnt` as counter
 *    c. Reverse iterate `S` to classify character and record bucket ends,
 *       clear bucket ends in `stack_cnt`
 * Output:
 *    * Bucket ends and classification are recorded in `S`
 *      S type characters are set to `(bucket_end << 1) | 1`;
 *      L type characters are set to `(bucket_start << 1)`
 *    * `stack_cnt` elements at bucket ends are cleared.
 */
static void rename_pat(std::span<int> S, std::span<int> bucket_cnt,
                       std::span<int> stack_cnt)
{
    int N = S.size();
    std::ranges::fill(bucket_cnt, 0);
    // Count bucket size
    for (auto v: S)
        bucket_cnt[v] += 1;
    assert(bucket_cnt[0] == 1); // Unique 0 character
    // Compute bucket boundaries
    int s = 1;
    for (int i = 1; i < N; i++) {
        s += bucket_cnt[i];
        bucket_cnt[i] = s;
    }
    assert(bucket_cnt[N - 1] == N);
    // Iterate from the end,
    // compute the type of the character (S/L)
    // and fill the corresponding slot in S with the start (for L type)
    // or end index (for S type) of the corresponding bucket.
    bool stype = true;
    int prev_v = 0;
    assert(S[N - 1] == 0);
    S[N - 1] = 1;
    stack_cnt[0] = 0;
    for (int i = N - 2; i >= 0; i--) {
        auto v = S[i];
        if (v != prev_v)
            stype = v < prev_v;
        if (stype) {
            auto bucket_end = bucket_cnt[v] - 1;
            stack_cnt[bucket_end] = 0;
            S[i] = (bucket_end << 1) | 1;
        }
        else {
            auto bucket_start = bucket_cnt[v - 1];
            stack_cnt[bucket_start] = 0;
            S[i] = bucket_start << 1;
        }
        prev_v = v;
    }
}

// With `SA` prefilled with LMS and -1,
// S type spots in `stack_cnt` set to non-negative number
// L type spot in `stack_cnt` set to `0`.
static void induce_sort(std::span<int> S, std::span<int> SA, std::span<int> stack_cnt)
{
    int N = S.size();
    for (auto v: SA) {
        if (v <= 0)
            continue;
        auto c = S[v - 1];
        if (c & 1)
            continue;
        auto bucket_start = c >> 1;
        auto cnt = stack_cnt[bucket_start];
        SA[bucket_start + cnt] = v - 1;
        stack_cnt[bucket_start] = cnt + 1;
    }
    for (int i = N - 1; i >= 0; i--) {
        auto v = SA[i];
        if (v <= 0)
            continue;
        auto c = S[v - 1];
        if ((c & 1) == 0)
            continue;
        auto bucket_end = c >> 1;
        auto cnt = stack_cnt[bucket_end];
        if (cnt > 0)
            cnt = 0;
        SA[bucket_end + cnt] = v - 1;
        stack_cnt[bucket_end] = cnt - 1;
    }
}

/**
 * Sort LMS prefix
 *    a. Clear `PA` (set to `-1`)
 *    b. Iterate `S` to find all LMS and fill them into the corresponding buckets ends
 *       in `PA`, using the bucket end in `stack_cnt` as stack counter
 *       (cleared in previous step)
 *    c. Iterate `PA` to fill in all L type characters into the buckets in `PA`,
 *       using the bucket start in `stack_cnt` as stack counter.
 *    d. Reverse iterate `PA` to fill in all S type characters into the buckets in `PA`,
 *       using the bucket end in `stack_cnt` as stack counter.
 *       This time, use negative values to count to avoid conflict
 *       with the LMS filling step. Now the LMS prefixes are sorted in `PA`.
 * Output:
 *    * `S` untouched, and still holds the character type/bucket ends info.
 *    * `PA` contains sorted LMS prefixes including LMS substrings
 *    * returns LMS count
 */
static int sort_lms_prefix(std::span<int> S, std::span<int> PA,
                           std::span<int> stack_cnt)
{
    int N = S.size();
    std::ranges::fill(PA, -1);
    bool stype = S[0] & 1;
    int lms_cnt = 0;
    for (int i = 1; i < N; i++) {
        auto c = S[i];
        bool new_stype = c & 1;
        bool lms = new_stype & !stype;
        stype = new_stype;
        if (!lms)
            continue;
        lms_cnt++;
        // S[i] is LMS, push it to the S type stack in the bucket
        auto bucket_end = c >> 1;
        auto cnt = stack_cnt[bucket_end];
        PA[bucket_end - cnt] = i;
        stack_cnt[bucket_end] = cnt + 1;
    }
    assert(lms_cnt * 2 <= N);
    induce_sort(S, PA, stack_cnt);
    return lms_cnt;
}

static bool lms_substr_equal(std::span<int> S, int prev_i, int self_i)
{
    int N = S.size();
    // Only the potentially smaller LMS could be the NUL byte.
    assert(self_i < N - 1);
    assert(prev_i <= N - 1);
    // Must not be the same
    assert(self_i != prev_i);
    // Must both starts as stype
    assert(S[prev_i] & 1);
    assert(S[self_i] & 1);
    if (prev_i == N - 1)
        return false;
    bool stype = true;
    for (int i = 0; ; i++) {
        assert(prev_i + i < N);
        assert(self_i + i < N);
        auto prev_c = S[prev_i + i];
        auto self_c = S[self_i + i];
        if (prev_c != self_c)
            return false;
        if (!(prev_c & 1)) {
            stype = false; // In the L region
        }
        else if (!stype) {
            // On a L -> S transition, that's the end
            return true;
        }
    }
}

/**
 * Create LMS substring array `S1`
 *    a. Iterate `PA`, for each LMS character `S[i]` found,
 *       compute the LMS substring id/name and store it to `ws[i]`.
 *       Keep track if there's duplicated LMS substring.
 *    b. Divide `PA` into to stacks one from the begining and one from the middle.
 *       Iterate `S`, for each LMS character `S[i]` found push `i` into the first stack
 *       and the LMS substring id `ws[i]` into the second stack.
 *       The second stack now contains `S1`.
 * Output:
 *    * `S` untouched, and still holds the character type/bucket ends info.
 *    * First half of `PA` holds the index (in `S` order) of LMS characters.
 *    * Second half of `PA` holds `S1`
 *    * returns number of unique lms
 */
static int create_s1(std::span<int> S, std::span<int> PA,
                     std::span<int> ws, int lms_cnt)
{
    int N = S.size();
    int N_2 = N / 2;
    int lms_name = -1;
    int next_pi = -1;
    int prev_i = 0;
    for (int pi = 0; pi < N; pi++) {
        auto i = PA[pi];
        assert(i >= 0);
        if (i <= 0)
            continue;
        bool left_stype = S[i - 1] & 1;
        bool self_stype = S[i] & 1;
        if (left_stype || !self_stype) // not LMS
            continue;
        if (pi != next_pi || !lms_substr_equal(S, prev_i, i))
            lms_name++;
        ws[i] = lms_name;
        prev_i = i;
        next_pi = pi + 1;
    }
    assert(lms_cnt <= N_2);
    assert(lms_name <= lms_cnt);
    auto lms_idx = PA.subspan(0, lms_cnt);
    auto S1 = PA.subspan(N_2, lms_cnt);
    int cnt = 0;
    bool stype = S[0] & 1;
    for (int i = 1; i < N; i++) {
        auto c = S[i];
        bool new_stype = c & 1;
        bool lms = new_stype & !stype;
        stype = new_stype;
        if (!lms)
            continue;
        lms_idx[cnt] = i;
        S1[cnt] = ws[i];
        cnt++;
    }
    assert(cnt == lms_cnt);
    return lms_name + 1;
}

/**
 * Create LMS suffix array `SA1`
 *    a. This is trivial if there's no duplicate in LMS substring.
 *       When there is duplicate, call the suffix array function
 *       recursively, with two halves of `SA` being the new `SA1` and `ws1`.
 * Output:
 *    * `S` untouched, and still holds the character type/bucket ends info.
 *    * First half of `ws` holds the index (in `S` order) of LMS characters.
 *    * First half of `SA` holds `SA1`
 */
static void sort_lms_suffix(std::span<int> S1, std::span<int> SA1,
                            std::span<int> ws1, int unique_lms)
{
    int lms_cnt = S1.size();
    assert(SA1.size() == lms_cnt);
    assert(ws1.size() == lms_cnt);
    if (unique_lms == lms_cnt) {
        order_to_rank(SA1, S1);
        return;
    }
    _get_suffix_array(SA1, S1, ws1);
}

/**
 * Sort suffix array using induction
 *    a. Replace each element of `SA1` in `SA` with the original LMS character
 *       index stored in `ws`
 *    b. Clear `ws` (to be used as stack counters)
 *    c. Clear the part of `SA` after where `SA1` is stored (fill with `-1`).
 *    d. Reverse iterate the LMS character in `SA1` (stored in `SA`) and place them
 *       in the corresponding bucket (at the end of the bucket) using the end counter
 *       that was cleared in `ws`. Replace the character in `SA1` with `-1`
 *       in the process.
 *    e. Iterate `SA` to fill in all L type characters into the buckets in `SA`,
 *       using the bucket start in `ws` as stack counter.
 *    f. Reverse iterate `SA` to fill in all S type characters into the buckets in `SA`,
 *       using the bucket end in `ws` as stack counter.
 *       This time, use negative values to count to avoid conflict
 *       with the LMS filling step. The sorting is done.
 * Output:
 *    * `SA` is sorted.
 */
static void sort_suffix(std::span<int> S, std::span<int> SA, std::span<int> ws,
                        int lms_cnt)
{
    auto lms_idx = ws.subspan(0, lms_cnt);
    auto SA1 = SA.subspan(0, lms_cnt);
    for (auto &v: SA1)
        v = lms_idx[v];
    auto stack_cnt = ws;
    std::ranges::fill(stack_cnt, 0);
    std::ranges::fill(SA.subspan(lms_cnt), -1);
    for (int sai = lms_cnt - 1; sai >= 0; sai--) {
        auto si = SA[sai];
        auto c = S[si];
        assert(c & 1);
        auto bucket_end = c >> 1;
        auto cnt = stack_cnt[bucket_end];
        auto new_sai = bucket_end - cnt;
        if (new_sai == sai)
            break;
        assert(new_sai > sai);
        SA[new_sai] = si;
        SA[sai] = -1;
        stack_cnt[bucket_end] = cnt + 1;
    }
    induce_sort(S, SA, stack_cnt);
}

/**
 * General procedure:
 * 1. Classify characters (L/S/LMS), construct bucket
 *    a. Clear `ws`
 *    b. Iterate `S` to count bucket size using `ws` as counter
 *    c. Reverse iterate `S` to classify character and record bucket ends,
 *       clear bucket ends in `SA`
 *  Output:
 *    * Bucket ends and classification are recorded in `S`
 *      S type characters are set to `(bucket_end << 1) | 1`;
 *      L type characters are set to `(bucket_start << 1)`
 *    * `SA` elements at bucket ends are cleared.
 * 2. Sort LMS prefix
 *    a. Clear `ws` (set to `-1`)
 *    b. Iterate `S` to find all LMS and fill them into the corresponding buckets ends
 *       in `ws`, using the bucket end in `SA` as stack counter
 *       (cleared in previous step)
 *    c. Iterate `ws` to fill in all L type characters into the buckets in `ws`,
 *       using the bucket start in `SA` as stack counter.
 *    d. Reverse iterate `ws` to fill in all S type characters into the buckets in `ws`,
 *       using the bucket end in `SA` as stack counter.
 *       This time, use negative values to count to avoid conflict
 *       with the LMS filling step. Now the LMS prefixes are sorted in `ws`.
 *  Output:
 *    * `S` untouched, and still holds the character type/bucket ends info.
 *    * `ws` contains sorted LMS prefixes including LMS substrings
 * 3. Create LMS substring array `S1`
 *    a. Iterate `ws`, for each LMS character `S[i]` found,
 *       compute the LMS substring id/name and store it to `SA[i]`.
 *       Keep track if there's duplicated LMS substring.
 *    b. Divide `ws` into to stacks one from the begining and one from the middle.
 *       Iterate `S`, for each LMS character `S[i]` found push `i` into the first stack
 *       and the LMS substring id `SA[i]` into the second stack.
 *       The second stack now contains `S1`.
 *  Output:
 *    * `S` untouched, and still holds the character type/bucket ends info.
 *    * First half of `ws` holds the index (in `S` order) of LMS characters.
 *    * Second half of `ws` holds `S1`
 * 4. Create LMS suffix array `SA1`
 *    a. This is trivial if there's no duplicate in LMS substring.
 *       When there is duplicate, call the suffix array function
 *       recursively, with two halves of `SA` being the new `SA1` and `ws1`.
 *  Output:
 *    * `S` untouched, and still holds the character type/bucket ends info.
 *    * First half of `ws` holds the index (in `S` order) of LMS characters.
 *    * First half of `SA` holds `SA1`
 * 5. Sort suffix array using induction
 *    a. Replace each element of `SA1` in `SA` with the original LMS character
 *       index stored in `ws`
 *    b. Clear `ws` (to be used as stack counters)
 *    c. Clear the part of `SA` after where `SA1` is stored (fill with `-1`).
 *    d. Reverse iterate the LMS character in `SA1` (stored in `SA`) and place them
 *       in the corresponding bucket (at the end of the bucket) using the end counter
 *       that was cleared in `ws`. Replace the character in `SA1` with `-1`
 *       in the process.
 *    e. Iterate `SA` to fill in all L type characters into the buckets in `SA`,
 *       using the bucket start in `ws` as stack counter.
 *    f. Reverse iterate `SA` to fill in all S type characters into the buckets in `SA`,
 *       using the bucket end in `ws` as stack counter.
 *       This time, use negative values to count to avoid conflict
 *       with the LMS filling step. The sorting is done.
 *  Output:
 *    * `SA` is sorted.
 */
__attribute__((flatten))
static void _get_suffix_array(std::span<int> SA, std::span<int> S,
                              std::span<int> ws)
{
    int N = S.size();
#ifndef NDEBUG
    assert(SA.size() == N);
    assert(ws.size() == N);
    for (int i = 0; i < N; i++) {
        auto v = S[i];
        assert(v < N);
        if (i == N - 1) {
            assert(v == 0);
        }
        else {
            assert(v > 0);
        }
    }
#endif
    rename_pat(S, ws, SA);
    auto lms_cnt = sort_lms_prefix(S, ws, SA);
    auto unique_lms = create_s1(S, ws, SA, lms_cnt);
    int N_2 = N / 2;
    sort_lms_suffix(ws.subspan(N_2, lms_cnt), SA.subspan(0, lms_cnt),
                    SA.subspan(N_2, lms_cnt), unique_lms);
    sort_suffix(S, SA, ws, lms_cnt);
}

__attribute__((visibility("protected")))
void get_height_array(std::span<int> height, std::span<int> S,
                      std::span<int> SA, std::span<int> RK)
{
    int N = S.size();
    if (N <= 2)
        return;
    assert(height.size() == N - 2);
    assert(SA.size() == N);
    assert(RK.size() == N);
    for (int i = 0, k = 0; i < N; i++) {
        auto rk = RK[i];
        if (rk <= 1) {
            k = 0;
            continue;
        }
        if (k)
            k -= 1;
        for (auto prev_i = SA[rk - 1]; S[i + k] == S[prev_i + k];)
            k++;
        height[rk - 2] = k;
    }
}

}
