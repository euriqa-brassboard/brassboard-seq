//

#include "utils.h"

#include <stdarg.h>
#include <stdlib.h>
#include <strings.h>

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

PyObject *BacktraceTracker::FrameInfo::get_traceback(PyObject *next)
{
    PyThreadState *tstate = PyThreadState_Get();
    PyObject *trace = nullptr;
    PyObject *globals = nullptr;
    PyObject *args = nullptr;

    globals = PyDict_New();
    args = PyTuple_New(4);
    if (!globals || !args)
        goto end;
    PyTuple_SET_ITEM(args, 0, Py_NewRef(next));
    if (auto frame = PyFrame_New(tstate, code, globals, nullptr))
        PyTuple_SET_ITEM(args, 1, (PyObject*)frame);
    else
        goto end;
    if (auto py_lasti = PyLong_FromLong(lasti))
        PyTuple_SET_ITEM(args, 2, py_lasti);
    else
        goto end;
    if (auto py_lineno = PyLong_FromLong(lineno))
        PyTuple_SET_ITEM(args, 3, py_lineno);
    else
        goto end;
    trace = traceback_new(&PyTraceBack_Type, args, nullptr);

end:
    Py_XDECREF(globals);
    Py_XDECREF(args);
    return trace;
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
    if (!traceback_new)
        return Py_NewRef(Py_None);
    auto it = traces.find(key);
    if (it == traces.end())
        return Py_NewRef(Py_None);
    auto &trace = it->second;
    PyObject *py_trace = Py_NewRef(Py_None);
    for (auto &info: trace) {
        auto new_trace = info.get_traceback(py_trace);
        if (!new_trace) {
            // Skip a frame if we couldn't construct it.
            PyErr_Clear();
            continue;
        }
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

void _bb_reraise(uintptr_t key)
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
    _bb_reraise(key);
}
