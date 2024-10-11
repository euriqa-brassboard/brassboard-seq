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

#include "utils.h"

#include <stdarg.h>
#include <stdlib.h>
#include <strings.h>

#include <algorithm>
#include <mutex>

namespace brassboard_seq {

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

#if PY_VERSION_HEX < 0x030b00f0

static inline PyCodeObject *pyframe_getcode(PyFrameObject *frame)
{
    return (PyCodeObject*)py_xnewref((PyObject*)frame->f_code);
}
static inline int pyframe_getlasti(PyFrameObject *frame)
{
    return frame->f_lasti;
}
static inline PyFrameObject *pyframe_getback(PyFrameObject *frame)
{
    return (PyFrameObject*)py_xnewref((PyObject*)frame->f_back);
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

PyObject *BacktraceTracker::FrameInfo::get_traceback(PyObject *next)
{
    PyThreadState *tstate = PyThreadState_Get();
    py_object globals(pydict_new());
    py_object args(pytuple_new(4));

    PyTuple_SET_ITEM(args.get(), 0, py_newref(next));
    PyTuple_SET_ITEM(args.get(), 1, (PyObject*)throw_if_not(
                         PyFrame_New(tstate, code, globals, nullptr)));
    PyTuple_SET_ITEM(args.get(), 2, pylong_from_long(lasti));
    PyTuple_SET_ITEM(args.get(), 3, pylong_from_long(lineno));
    return throw_if_not(traceback_new(&PyTraceBack_Type, args, nullptr));
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
            Py_DECREF(frame);
        trace.push_back({new_frame});
        frame = new_frame;
        frame_need_free = true;
    }
    if (frame_need_free) {
        Py_DECREF(frame);
    }
}

PyObject *BacktraceTracker::get_backtrace(uintptr_t key)
{
    assert(traceback_new);
    auto it = traces.find(key);
    if (it == traces.end())
        return nullptr;
    auto &trace = it->second;
    PyObject *py_trace = nullptr;
    for (auto &info: trace) {
        try {
            auto new_trace = info.get_traceback(py_trace ? py_trace : Py_None);
            Py_XDECREF(py_trace);
            py_trace = new_trace;
        }
        catch (...) {
            // Skip a frame if we couldn't construct it.
            PyErr_Clear();
        }
    }
    return py_trace;
}

static PyObject *combine_traceback(PyObject *old_tb, PyObject *tb)
{
    // both tb and old_tb are owning references, returning an owning reference.
    if (!old_tb)
        return tb;
    if (tb) {
        auto last_tb = (PyTracebackObject*)old_tb;
        while (last_tb->tb_next)
            last_tb = last_tb->tb_next;
        last_tb->tb_next = (PyTracebackObject*)tb;
    }
    return old_tb;
}

static inline PyObject *get_global_backtrace(uintptr_t key)
{
    if (BacktraceTracker::global_tracker)
        return BacktraceTracker::global_tracker->get_backtrace(key);
    return nullptr;
}

[[noreturn]] void throw0()
{
    throw 0;
}

void _bb_raise(PyObject *exc, uintptr_t key)
{
    auto type = (PyObject*)Py_TYPE(exc);
    PyErr_Restore(py_newref(type), py_newref(exc),
                  combine_traceback(PyException_GetTraceback(exc),
                                    get_global_backtrace(key)));
}

void bb_reraise(uintptr_t key)
{
    PyObject *exc, *type, *old_tb;
    PyErr_Fetch(&type, &exc, &old_tb);
    PyErr_Restore(type, exc, combine_traceback(old_tb, get_global_backtrace(key)));
}

void _bb_err_format(PyObject *exc, uintptr_t key, const char *format, ...)
{
    // This is slightly less efficient but much simpler to implement.
    va_list vargs;
    va_start(vargs, format);
    PyErr_FormatV(exc, format, vargs);
    va_end(vargs);
    bb_reraise(key);
}

[[noreturn]] void bb_rethrow(uintptr_t key)
{
    bb_reraise(key);
    throw0();
}

[[noreturn]] void bb_throw_format(PyObject *exc, uintptr_t key,
                                  const char *format, ...)
{
    // This is slightly less efficient but much simpler to implement.
    va_list vargs;
    va_start(vargs, format);
    PyErr_FormatV(exc, format, vargs);
    va_end(vargs);
    bb_rethrow(key);
}

[[noreturn]] void py_throw_format(PyObject *exc, const char *format, ...)
{
    // This is slightly less efficient but much simpler to implement.
    va_list vargs;
    va_start(vargs, format);
    PyErr_FormatV(exc, format, vargs);
    va_end(vargs);
    bb_rethrow(uintptr_t(-1));
}

// We will leak these objects.
// Otherwise, the destructor may be called after the libpython is already shut down.
PyObject *pyfloat_m1(PyFloat_FromDouble(-1));
PyObject *pyfloat_m0_5(PyFloat_FromDouble(-0.5));
PyObject *pyfloat_0(PyFloat_FromDouble(0));
PyObject *pyfloat_0_5(PyFloat_FromDouble(0.5));
PyObject *pyfloat_1(PyFloat_FromDouble(1));

PyObject *pytuple_append1(PyObject *tuple, PyObject *obj)
{
    Py_ssize_t nele = PyTuple_GET_SIZE(tuple);
    py_object res(pytuple_new(nele + 1));
    for (Py_ssize_t i = 0; i < nele; i++)
        PyTuple_SET_ITEM(res.get(), i, py_newref(PyTuple_GET_ITEM(tuple, i)));
    PyTuple_SET_ITEM(res.get(), nele, py_newref(obj));
    return res.release();
}

static PyObject *_pydict_deepcopy(PyObject *d)
{
    py_object res(pydict_new());

    PyObject *key;
    PyObject *value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(d, &pos, &key, &value)) {
        if (!PyDict_Check(value)) {
            throw_if(PyDict_SetItem(res.get(), key, value));
            continue;
        }
        py_object new_value(_pydict_deepcopy(value));
        throw_if(PyDict_SetItem(res.get(), key, new_value.get()));
    }
    return res.release();
}

PyObject *pydict_deepcopy(PyObject *d)
{
    if (!PyDict_Check(d))
        return py_newref(d);
    return _pydict_deepcopy(d);
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

void get_height_array(std::span<int> height, std::span<int> S,
                      std::span<int> SA, std::span<int> RK)
{
    int N = S.size();
    assert(height.size() == N);
    assert(SA.size() == N);
    assert(RK.size() == N);
    for (int i = 0, k = 0; i < N; i++) {
        auto rk = RK[i];
        if (rk == 0) {
            height[0] = 0;
            continue;
        }
        if (k)
            k -= 1;
        for (auto prev_i = SA[rk - 1]; S[i + k] == S[prev_i + k];)
            k++;
        height[rk] = k;
    }
}

static std::once_flag init_flag;

namespace rtval {
void init(); // Too lazy to create a private header just for this...
}

void init_library()
{
    std::call_once(init_flag, [] {
        rtval::init();
    });
}

}
