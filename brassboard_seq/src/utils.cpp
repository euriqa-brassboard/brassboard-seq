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

#include <array>
#include <mutex>

#include "numpy/arrayobject.h"

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

PyObject *BacktraceTracker::FrameInfo::get_traceback(PyObject *next) try
{
    PyThreadState *tstate = PyThreadState_Get();
    py_object globals(throw_if_not(PyDict_New()));
    py_object args(throw_if_not(PyTuple_New(4)));

    PyTuple_SET_ITEM(args.get(), 0, py_newref(next));
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
    py_object res(throw_if_not(PyTuple_New(nele + 1)));
    for (Py_ssize_t i = 0; i < nele; i++)
        PyTuple_SET_ITEM(res.get(), i, py_newref(PyTuple_GET_ITEM(tuple, i)));
    PyTuple_SET_ITEM(res.get(), nele, py_newref(obj));
    return res.release();
}

static PyObject *_pydict_deepcopy(PyObject *d)
{
    py_object res(throw_if_not(PyDict_New()));

    PyObject *key;
    PyObject *value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(d, &pos, &key, &value)) {
        if (!PyDict_Check(value)) {
            throw_if(PyDict_SetItem(res.get(), key, value) < 0);
            continue;
        }
        py_object new_value(_pydict_deepcopy(value));
        throw_if(PyDict_SetItem(res.get(), key, new_value.get()) < 0);
    }
    return res.release();
}

PyObject *pydict_deepcopy(PyObject *d)
{
    if (!PyDict_Check(d))
        return py_newref(d);
    return _pydict_deepcopy(d);
}

static std::once_flag init_numpy;

void init_library()
{
    std::call_once(init_numpy, [] { _import_array(); });
}

namespace rtval {

__attribute__((flatten))
void _rt_eval_cache(_RuntimeValue *self, unsigned age, py_object &pyage)
{
    if (self->age == age)
        return;

    // Take the reference from the argument
    auto set_cache = [&] (TagVal v) {
        assert(v.type == self->datatype);
        self->cache_val = v.val;
        self->cache_err = v.err;
        self->age = age;
    };
    auto set_cache_py = [&] (PyObject *obj) {
        throw_if_not(obj);
        set_cache(TagVal::from_py(obj).convert(self->datatype));
        Py_DECREF(obj);
    };

    auto type = self->type_;
    switch (type) {
    case Arg:
        PyErr_Format(PyExc_ValueError, "Cannot evaluate unknown argument");
        throw 0;
    case Const:
        return;
    case Extern:
        set_cache_py(_PyObject_Vectorcall(self->cb_arg2, nullptr, 0, nullptr));
        return;
    case ExternAge: {
        if (!pyage)
            pyage.reset(throw_if_not(PyLong_FromLong(age)));
        PyObject *args[] = { pyage.get() };
        set_cache_py(_PyObject_Vectorcall(self->cb_arg2, args, 1, nullptr));
        return;
    }
    default:
        break;
    }

    auto rtarg0 = self->arg0;
    _rt_eval_cache(rtarg0, age, pyage);
    auto arg0 = rtval_cache(rtarg0);
    auto eval1 = [&] (auto op_cls) {
        if (arg0.err != EvalError::NoError) {
            set_cache({ self->datatype, arg0.err });
        }
        else {
            set_cache(op_cls.generic_eval(arg0));
        }
    };

    switch (type) {
#define HANDLE_UNARY(op) case op: eval1(op##_op()); return
        HANDLE_UNARY(Not);
        HANDLE_UNARY(Bool);
        HANDLE_UNARY(Abs);
        HANDLE_UNARY(Ceil);
        HANDLE_UNARY(Floor);
        HANDLE_UNARY(Exp);
        HANDLE_UNARY(Expm1);
        HANDLE_UNARY(Log);
        HANDLE_UNARY(Log1p);
        HANDLE_UNARY(Log2);
        HANDLE_UNARY(Log10);
        HANDLE_UNARY(Sqrt);
        HANDLE_UNARY(Asin);
        HANDLE_UNARY(Acos);
        HANDLE_UNARY(Atan);
        HANDLE_UNARY(Asinh);
        HANDLE_UNARY(Acosh);
        HANDLE_UNARY(Atanh);
        HANDLE_UNARY(Sin);
        HANDLE_UNARY(Cos);
        HANDLE_UNARY(Tan);
        HANDLE_UNARY(Sinh);
        HANDLE_UNARY(Cosh);
        HANDLE_UNARY(Tanh);
        HANDLE_UNARY(Rint);
        HANDLE_UNARY(Int64);
#undef HANDLE_UNARY
    default:
        break;
    }

    auto rtarg1 = self->arg1;
    if (type == Select) {
        auto rtarg2 = (_RuntimeValue*)self->cb_arg2;
        auto rtres = arg0.template get<bool>() ? rtarg1 : rtarg2;
        _rt_eval_cache(rtres, age, pyage);
        set_cache(rtval_cache(rtres).convert(self->datatype));
        return;
    }
    _rt_eval_cache(rtarg1, age, pyage);
    auto arg1 = rtval_cache(rtarg1);

    auto eval2 = [&] (auto op_cls) {
        if (auto err = combine_error(arg0.err, arg1.err); err != EvalError::NoError) {
            set_cache({ self->datatype, err });
        }
        else {
            set_cache(op_cls.generic_eval(arg0, arg1));
        }
    };

    switch (type) {
#define HANDLE_BINARY(op) case op: eval2(op##_op()); return
        HANDLE_BINARY(Add);
        HANDLE_BINARY(Sub);
        HANDLE_BINARY(Mul);
        HANDLE_BINARY(Div);
        HANDLE_BINARY(Pow);
        HANDLE_BINARY(Mod);
        HANDLE_BINARY(And);
        HANDLE_BINARY(Or);
        HANDLE_BINARY(Xor);
        HANDLE_BINARY(CmpLT);
        HANDLE_BINARY(CmpGT);
        HANDLE_BINARY(CmpLE);
        HANDLE_BINARY(CmpGE);
        HANDLE_BINARY(CmpNE);
        HANDLE_BINARY(CmpEQ);
        HANDLE_BINARY(Hypot);
        HANDLE_BINARY(Atan2);
        HANDLE_BINARY(Max);
        HANDLE_BINARY(Min);
#undef HANDLE_BINARY
    default:
        PyErr_Format(PyExc_ValueError, "Unknown value type");
        throw 0;
    }
}

static inline bool is_numpy_int(PyObject *value)
{
    if (PyArray_IsScalar(value, Integer))
        return true;
    return PyArray_IsZeroDim(value) && PyArray_ISINTEGER((PyArrayObject*)value);
}

TagVal TagVal::from_py(PyObject *value)
{
    if (value == Py_True)
        return true;
    if (value == Py_False)
        return false;
    if (PyLong_Check(value) || is_numpy_int(value)) {
        auto val = PyLong_AsLongLong(value);
        throw_if(val == -1 && PyErr_Occurred());
        return TagVal(val);
    }
    auto val = PyFloat_AsDouble(value);
    throw_if(val == -1 && PyErr_Occurred());
    return TagVal(val);
}

__attribute__((flatten, noinline, visibility("protected")))
std::pair<EvalError,GenVal> interpret_func(const int *code, GenVal *data,
                                           EvalError *errors)
{
#define GEN_UNI_OP(f, t1)                                       \
    int((char*)&&f##_op_##t1##_label - (char*)&&return_label),
#define GEN_BIN_OP(f, t1, t2)                                           \
    int((char*)&&f##_op_##t1##_##t2##_label - (char*)&&return_label),
#define GEN_SELECT_OP(t2, t3)                                           \
    int((char*)&&Select_op_##t2##_##t3##_label - (char*)&&return_label),
    static int const label_offsets[]
        asm(".L_ZN14brassboard_seq5rtval13label_offsetsE")
        __attribute__((used)) = {
#include "rtval_interp.h"
    };
#undef GEN_UNI_OP
#undef GEN_BIN_OP
#undef GEN_SELECT_OP

    // Making this variable `const` messes up clang's codegen for lambda
    // Ref https://github.com/llvm/llvm-project/issues/103309
    char *base_addr = (char*)&&return_label;

    auto pop_operand = [&] {
        auto res = *code;
        code += 1;
        return std::make_pair(&errors[res], &data[res]);
    };
    auto pop_label = [&] {
        auto res = *code;
        code += 1;
        return (void*)(base_addr + res);
    };

    goto *pop_label();

return_label: {
        auto [eo, out] = pop_operand();
        return {*eo, *out};
    }

#define GEN_UNI_OP(f, t1)                                               \
    f##_op_##t1##_label: {                                              \
        auto [eo, out] = pop_operand();                                 \
        auto [e1, in1] = pop_operand();                                 \
        constexpr auto out_dt = f##_op::return_type(DataType::t1);      \
        using T1 = data_type_t<DataType::t1>;                           \
        using Tout = data_type_t<out_dt>;                               \
        auto v = in1->get<T1>();                                        \
        auto res = f##_op::template eval_err<Tout,T1>(v);               \
        *eo = combine_error(*e1, res.err);                              \
        *out = res.val;                                                 \
    }                                                                   \
    goto *pop_label();
#define GEN_BIN_OP(f, t1, t2)                                           \
    f##_op_##t1##_##t2##_label: {                                       \
        auto [eo, out] = pop_operand();                                 \
        auto [e1, in1] = pop_operand();                                 \
        auto [e2, in2] = pop_operand();                                 \
        constexpr auto out_dt = f##_op::return_type(DataType::t1, DataType::t2); \
        using T1 = data_type_t<DataType::t1>;                           \
        using T2 = data_type_t<DataType::t2>;                           \
        using Tout = data_type_t<out_dt>;                               \
        auto v1 = in1->get<T1>();                                       \
        auto v2 = in2->get<T2>();                                       \
        auto res = f##_op::template eval_err<Tout,T1,T2>(v1, v2);       \
        *eo = combine_error(*e1, *e2, res.err);                         \
        *out = res.val;                                                 \
    }                                                                   \
    goto *pop_label();
#define GEN_SELECT_OP(t1, t2)                                           \
    Select_op_##t1##_##t2##_label: {                                    \
        auto [eo, out] = pop_operand();                                 \
        auto [e0, in0] = pop_operand();                                 \
        auto [e1, in1] = pop_operand();                                 \
        auto [e2, in2] = pop_operand();                                 \
        bool b = bool(in0->i64_val);                                    \
        constexpr auto out_dt = promote_type(DataType::t1, DataType::t2); \
        using T1 = data_type_t<DataType::t1>;                           \
        using T2 = data_type_t<DataType::t2>;                           \
        using Tout = data_type_t<out_dt>;                               \
        auto v1 = Tout(in1->get<T1>());                                 \
        auto v2 = Tout(in2->get<T2>());                                 \
        out->get<Tout>() = b ? v1 : v2;                                 \
        *eo = combine_error(*e0, b ? *e1 : *e2);                        \
    }                                                                   \
    goto *pop_label();
#include "rtval_interp.h"
#undef GEN_UNI_OP
#undef GEN_BIN_OP
#undef GEN_SELECT_OP
}

static inline constexpr
int get_label_id(ValueType f, DataType t1, DataType t2=DataType(0))
{
    return f * 9 + int(t1) * 3 + int(t2);
}

static const auto interp_label_offsets = [] {
    auto get_size = [] {
        int res = 0;
#define GEN_UNI_OP(f, t1)                                       \
        res = std::max(res, get_label_id(f, DataType::t1));
#define GEN_BIN_OP(f, t1, t2)                                           \
        res = std::max(res, get_label_id(f, DataType::t1, DataType::t2));
#define GEN_SELECT_OP(t2, t3)                                           \
        res = std::max(res, get_label_id(Select, DataType::t2, DataType::t3));
#include "rtval_interp.h"
#undef GEN_UNI_OP
#undef GEN_BIN_OP
#undef GEN_SELECT_OP
        return res + 1;
    };
    // Call the function once to guarantee that the static variable is initialized
    {
        const int code[] = {0, 0};
        GenVal vals[] = {{}};
        EvalError errors[] = {{}};
        interpret_func(code, vals, errors);
    }
    extern const int __attribute__((visibility("internal"))) label_offsets[]
        asm(".L_ZN14brassboard_seq5rtval13label_offsetsE");
    std::array<int,get_size()> res{};
    uint16_t idx = 0;
#define GEN_UNI_OP(f, t1)                                        \
    res[get_label_id(f, DataType::t1)] = label_offsets[idx];     \
    idx++;
#define GEN_BIN_OP(f, t1, t2)                                           \
    res[get_label_id(f, DataType::t1, DataType::t2)] = label_offsets[idx]; \
    idx++;
#define GEN_SELECT_OP(t2, t3)                                           \
    res[get_label_id(Select, DataType::t2, DataType::t3)] = label_offsets[idx]; \
    idx++;
#include "rtval_interp.h"
#undef GEN_UNI_OP
#undef GEN_BIN_OP
#undef GEN_SELECT_OP
    return res;
} ();

// For ifelse/select, the t1 t2 below is actually t2, t3 since the actual t1 isn't used.
int get_label_offset(ValueType op, DataType t1, DataType t2)
{
    return interp_label_offsets[get_label_id(op, t1, t2)];
}

void InterpFunction::_set_value(_RuntimeValue *value, std::vector<DataType> &args)
{
    int nargs = args.size();
    code.clear();
    data.clear();
    data.resize(nargs, GenVal{ .i64_val = 0 });
    rt_vals.clear();
    Builder builder{ nargs, args };
    auto &info = visit_value(value, builder);
    if (!info.dynamic)
        ensure_index(info, builder);
    rt_vals.resize(data.size(), 0);
    errors.resize(data.size());
    for (auto &[val, info]: builder.value_infos) {
        if (!info.dynamic && info.idx >= 0) {
            // Record the constants that needs to be filled in before evaluation.
            rt_vals[info.idx] = val;
        }
    }
    ret_type = info.val.type;
    // Insert return instruction to code.
    code.push_back(0);
    code.push_back(info.idx);
}

__attribute__((visibility("internal")))
inline InterpFunction::Builder::ValueInfo&
InterpFunction::visit_value(_RuntimeValue *value, Builder &builder)
{
    auto &info = builder.value_infos[value];
    if (info.inited)
        return info;

    auto type = value->type_;
    switch (type) {
    case Const: {
        info.is_const = true;
        info.inited = true;
        info.val = rtval_cache(value);
        return info;
    }
    case Arg: {
        auto v = PyLong_AsLong(value->cb_arg2);
        if (v < 0 || v >= builder.nargs) {
            if (!PyErr_Occurred())
                PyErr_Format(PyExc_IndexError,
                             "Argument index out of bound: %ld.", v);
            throw 0;
        }
        info.val.type = builder.types[v];
        info.dynamic = true;
        info.inited = true;
        info.idx = v;
        return info;
    }
    case Extern:
    case ExternAge: {
        // Hard coded for now.
        info.val.type = DataType::Float64;
        info.inited = true;
        return info;
    }
    default:
        break;
    }

    auto rtarg0 = value->arg0;
    auto &arg0_info = visit_value(rtarg0, builder);
    auto handle_unary = [&] (DataType ret_type) -> auto& {
        info.val.type = ret_type;
        info.inited = true;
        info.dynamic = arg0_info.dynamic;
        if (info.dynamic) {
            auto arg0_idx = arg0_info.idx;
            assert(arg0_idx >= 0);
            auto idx = ensure_index(info, builder);
            code.push_back(get_label_offset(type, arg0_info.val.type, DataType::Bool));
            code.push_back(idx);
            code.push_back(arg0_idx);
        }
        return info;
    };
#define HANDLE_UNARY(op)                                                \
    case op: return handle_unary(op##_op::return_type(arg0_info.val.type))

    switch (type) {
    HANDLE_UNARY(Not);
    HANDLE_UNARY(Bool);
    HANDLE_UNARY(Abs);
    HANDLE_UNARY(Ceil);
    HANDLE_UNARY(Floor);
    HANDLE_UNARY(Exp);
    HANDLE_UNARY(Expm1);
    HANDLE_UNARY(Log);
    HANDLE_UNARY(Log1p);
    HANDLE_UNARY(Log2);
    HANDLE_UNARY(Log10);
    HANDLE_UNARY(Sqrt);
    HANDLE_UNARY(Asin);
    HANDLE_UNARY(Acos);
    HANDLE_UNARY(Atan);
    HANDLE_UNARY(Asinh);
    HANDLE_UNARY(Acosh);
    HANDLE_UNARY(Atanh);
    HANDLE_UNARY(Sin);
    HANDLE_UNARY(Cos);
    HANDLE_UNARY(Tan);
    HANDLE_UNARY(Sinh);
    HANDLE_UNARY(Cosh);
    HANDLE_UNARY(Tanh);
    HANDLE_UNARY(Rint);
    HANDLE_UNARY(Int64);
#undef HANDLE_UNARY
    default:
        break;
    }

    auto rtarg1 = value->arg1;
    auto &arg1_info = visit_value(rtarg1, builder);
    if (type == Select) {
        auto rtarg2 = (_RuntimeValue*)value->cb_arg2;
        auto &arg2_info = visit_value(rtarg2, builder);
        info.val.type = promote_type(arg1_info.val.type, arg2_info.val.type);
        info.inited = true;
        info.dynamic = arg0_info.dynamic || arg1_info.dynamic || arg2_info.dynamic;
        if (info.dynamic) {
            auto arg0_idx = ensure_index(arg0_info, builder);
            auto arg1_idx = ensure_index(arg1_info, builder);
            auto arg2_idx = ensure_index(arg2_info, builder);
            auto idx = ensure_index(info, builder);
            code.push_back(get_label_offset(Select, arg1_info.val.type,
                                            arg2_info.val.type));
            code.push_back(idx);
            code.push_back(arg0_idx);
            code.push_back(arg1_idx);
            code.push_back(arg2_idx);
        }
        return info;
    }

    auto handle_binary = [&] (DataType ret_type) -> auto& {
        info.val.type = ret_type;
        info.inited = true;
        info.dynamic = arg0_info.dynamic || arg1_info.dynamic;
        if (info.dynamic) {
            auto arg0_idx = ensure_index(arg0_info, builder);
            auto arg1_idx = ensure_index(arg1_info, builder);
            auto idx = ensure_index(info, builder);
            code.push_back(get_label_offset(type, arg0_info.val.type,
                                            arg1_info.val.type));
            code.push_back(idx);
            code.push_back(arg0_idx);
            code.push_back(arg1_idx);
        }
        return info;
    };
#define HANDLE_BINARY(op)                                               \
    case op: return handle_binary(op##_op::return_type(arg0_info.val.type, \
                                                       arg1_info.val.type))

    switch (type) {
    HANDLE_BINARY(Add);
    HANDLE_BINARY(Sub);
    HANDLE_BINARY(Mul);
    HANDLE_BINARY(Div);
    HANDLE_BINARY(Pow);
    HANDLE_BINARY(Mod);
    HANDLE_BINARY(And);
    HANDLE_BINARY(Or);
    HANDLE_BINARY(Xor);
    HANDLE_BINARY(CmpLT);
    HANDLE_BINARY(CmpGT);
    HANDLE_BINARY(CmpLE);
    HANDLE_BINARY(CmpGE);
    HANDLE_BINARY(CmpNE);
    HANDLE_BINARY(CmpEQ);
    HANDLE_BINARY(Hypot);
    HANDLE_BINARY(Atan2);
    HANDLE_BINARY(Max);
    HANDLE_BINARY(Min);
    default:
        PyErr_Format(PyExc_ValueError, "Unknown value type");
        throw 0;
    }
}

void InterpFunction::_eval_all(unsigned age, py_object &pyage)
{
    for (size_t i = 0; i < rt_vals.size(); i++) {
        auto rt_val = (_RuntimeValue*)rt_vals[i];
        if (!rt_val) {
            errors[i] = EvalError::NoError;
            continue;
        }
        _rt_eval_cache(rt_val, age, pyage);
        data[i] = rt_val->cache_val;
        errors[i] = rt_val->cache_err;
    }
}

}

}
