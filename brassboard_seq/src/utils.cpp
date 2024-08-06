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

#if PY_VERSION_HEX < 0x030a00f0

static inline PyObject *Py_NewRef(PyObject *obj)
{
    Py_INCREF(obj);
    return obj;
}

static inline PyObject *Py_XNewRef(PyObject *obj)
{
    Py_XINCREF(obj);
    return obj;
}

#endif

#if PY_VERSION_HEX < 0x030b00f0

static inline PyCodeObject *pyframe_getcode(PyFrameObject *frame)
{
    return (PyCodeObject*)Py_XNewRef((PyObject*)frame->f_code);
}
static inline int pyframe_getlasti(PyFrameObject *frame)
{
    return frame->f_lasti;
}
static inline PyFrameObject *pyframe_getback(PyFrameObject *frame)
{
    return (PyFrameObject*)Py_XNewRef((PyObject*)frame->f_back);
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

PyObject *BacktraceTracker::FrameInfo::get_traceback(PyObject *next) try
{
    PyThreadState *tstate = PyThreadState_Get();
    py_object globals(throw_if_not(PyDict_New()));
    py_object args(throw_if_not(PyTuple_New(4)));

    PyTuple_SET_ITEM(args.get(), 0, Py_NewRef(next));
    PyTuple_SET_ITEM(args.get(), 1, (PyObject*)throw_if_not(
                         PyFrame_New(tstate, code, globals, nullptr)));
    PyTuple_SET_ITEM(args.get(), 2, throw_if_not(PyLong_FromLong(lasti)));
    PyTuple_SET_ITEM(args.get(), 3, throw_if_not(PyLong_FromLong(lineno)));
    return traceback_new(&PyTraceBack_Type, args, nullptr);
}
catch (...) {
    return nullptr;
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
        auto new_trace = info.get_traceback(py_trace ? py_trace : Py_None);
        if (!new_trace) {
            // Skip a frame if we couldn't construct it.
            PyErr_Clear();
            continue;
        }
        Py_XDECREF(py_trace);
        py_trace = new_trace;
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

void _bb_raise(PyObject *exc, uintptr_t key)
{
    auto type = (PyObject*)Py_TYPE(exc);
    PyErr_Restore(Py_NewRef(type), Py_NewRef(exc),
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

// We will leak these objects.
// Otherwise, the destructor may be called after the libpython is already shut down.
PyObject *pyfloat_m1(PyFloat_FromDouble(-1));
PyObject *pyfloat_m0_5(PyFloat_FromDouble(-0.5));
PyObject *pyfloat_0(PyFloat_FromDouble(0));
PyObject *pyfloat_0_5(PyFloat_FromDouble(0.5));
PyObject *pyfloat_1(PyFloat_FromDouble(1));

}
