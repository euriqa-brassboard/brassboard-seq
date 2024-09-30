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

#include "rtval.h"

namespace brassboard_seq::rtval {

static inline TagVal tagval_add_or_sub(TagVal v1, TagVal v2, bool issub)
{
    assert(v1.err == EvalError::NoError);
    assert(v2.err == EvalError::NoError);
    return (issub ? Sub_op::generic_eval(v1, v2) : Add_op::generic_eval(v1, v2));
}

template<typename RuntimeValue>
static inline __attribute__((returns_nonnull)) RuntimeValue*
_new_expr2_wrap1(PyObject *RTValueType, ValueType type,
                 PyObject *arg0, PyObject *arg1, RuntimeValue*)
{
    py_object rtarg0;
    py_object rtarg1;
    if (Py_TYPE(arg0) != (PyTypeObject*)RTValueType) {
        rtarg0.reset((PyObject*)new_const(RTValueType, arg0, (RuntimeValue*)nullptr));
        rtarg1.reset(py_newref(arg1));
    }
    else {
        if (Py_TYPE(arg1) == (PyTypeObject*)RTValueType) {
            rtarg1.reset(py_newref(arg1));
        }
        else {
            rtarg1.reset((PyObject*)new_const(RTValueType, arg1,
                                              (RuntimeValue*)nullptr));
        }
        rtarg0.reset(py_newref(arg0));
    }
    auto datatype = binary_return_type(type, ((RuntimeValue*)rtarg0.get())->datatype,
                                       ((RuntimeValue*)rtarg1.get())->datatype);
    auto o = throw_if_not(PyType_GenericAlloc((PyTypeObject*)RTValueType, 0));
    auto self = (RuntimeValue*)o;
    self->datatype = datatype;
    // self->cache_err = EvalError::NoError;
    // self->cache_val = { .i64_val = 0 };
    self->type_ = type;
    self->age = (unsigned)-1;
    self->arg0 = (RuntimeValue*)rtarg0.release();
    self->arg1 = (RuntimeValue*)rtarg1.release();
    self->cb_arg2 = py_newref(Py_None);
    return self;
}

template<typename RuntimeValue>
static inline __attribute__((returns_nonnull)) RuntimeValue*
_wrap_rtval(PyObject *RTValueType, PyObject *v, RuntimeValue*)
{
    if (Py_TYPE(v) == (PyTypeObject*)RTValueType)
        return (RuntimeValue*)py_newref(v);
    return new_const(RTValueType, v, (RuntimeValue*)nullptr);
}

template<typename RuntimeValue>
static inline __attribute__((returns_nonnull)) RuntimeValue*
_new_select(PyObject *RTValueType, RuntimeValue *arg0,
            PyObject *arg1, PyObject *arg2)
{
    py_object rtarg1((PyObject*)_wrap_rtval(RTValueType, arg1, (RuntimeValue*)nullptr));
    py_object rtarg2((PyObject*)_wrap_rtval(RTValueType, arg2, (RuntimeValue*)nullptr));
    auto datatype = promote_type(((RuntimeValue*)rtarg1.get())->datatype,
                                 ((RuntimeValue*)rtarg2.get())->datatype);
    auto o = throw_if_not(PyType_GenericAlloc((PyTypeObject*)RTValueType, 0));
    auto self = (RuntimeValue*)o;
    self->datatype = datatype;
    // self->cache_err = EvalError::NoError;
    // self->cache_val = { .i64_val = 0 };
    self->type_ = Select;
    self->age = (unsigned)-1;
    self->arg0 = (RuntimeValue*)py_newref((PyObject*)arg0);
    self->arg1 = (RuntimeValue*)rtarg1.release();
    self->cb_arg2 = rtarg2.release();
    return self;
}

template<typename RuntimeValue>
inline void InterpFunction::set_value(RuntimeValue *value, std::vector<DataType> &args)
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

template<typename RuntimeValue>
__attribute__((visibility("internal")))
inline InterpFunction::Builder::ValueInfo&
InterpFunction::visit_value(RuntimeValue *value, Builder &builder)
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
        auto rtarg2 = (RuntimeValue*)value->cb_arg2;
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

template<typename RuntimeValue>
inline void InterpFunction::eval_all(unsigned age, py_object &pyage,
                                     RuntimeValue*)
{
    for (size_t i = 0; i < rt_vals.size(); i++) {
        auto rt_val = (RuntimeValue*)rt_vals[i];
        if (!rt_val) {
            errors[i] = EvalError::NoError;
            continue;
        }
        rt_eval_cache(rt_val, age, pyage);
        data[i] = rt_val->cache_val;
        errors[i] = rt_val->cache_err;
    }
}

}
