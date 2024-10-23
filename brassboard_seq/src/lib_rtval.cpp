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

#include "lib_rtval.h"

#include <array>

#include "numpy/arrayobject.h"

namespace brassboard_seq::rtval {

__attribute__((visibility("protected")))
PyTypeObject *RTVal_Type;

__attribute__((returns_nonnull,visibility("protected"))) _RuntimeValue*
_new_cb_arg2(ValueType type, PyObject *cb_arg2, PyObject *ty)
{
    auto datatype = pytype_to_datatype(ty);
    auto o = pytype_genericalloc(RTVal_Type);
    auto self = (_RuntimeValue*)o;
    self->datatype = datatype;
    // self->cache_err = EvalError::NoError;
    // self->cache_val = { .i64_val = 0 };
    self->type_ = type;
    self->age = (unsigned)-1;
    self->arg0 = (_RuntimeValue*)py_newref(Py_None);
    self->arg1 = (_RuntimeValue*)py_newref(Py_None);
    self->cb_arg2 = py_newref(cb_arg2);
    return self;
}

__attribute__((returns_nonnull,visibility("protected"))) _RuntimeValue*
_new_expr1(ValueType type, _RuntimeValue *arg0)
{
    auto o = pytype_genericalloc(RTVal_Type);
    auto datatype = unary_return_type(type, arg0->datatype);
    auto self = (_RuntimeValue*)o;
    self->datatype = datatype;
    // self->cache_err = EvalError::NoError;
    // self->cache_val = { .i64_val = 0 };
    self->type_ = type;
    self->age = (unsigned)-1;
    self->arg0 = py_newref(arg0);
    self->arg1 = (_RuntimeValue*)py_newref(Py_None);
    self->cb_arg2 = py_newref(Py_None);
    return self;
}

__attribute__((returns_nonnull,visibility("protected"))) _RuntimeValue*
_new_expr2(ValueType type, _RuntimeValue *arg0, _RuntimeValue *arg1)
{
    auto o = pytype_genericalloc(RTVal_Type);
    auto datatype = binary_return_type(type, arg0->datatype, arg1->datatype);
    auto self = (_RuntimeValue*)o;
    self->datatype = datatype;
    // self->cache_err = EvalError::NoError;
    // self->cache_val = { .i64_val = 0 };
    self->type_ = type;
    self->age = (unsigned)-1;
    self->arg0 = py_newref(arg0);
    self->arg1 = py_newref(arg1);
    self->cb_arg2 = py_newref(Py_None);
    return self;
}

__attribute__((returns_nonnull,visibility("protected"))) _RuntimeValue*
_new_const(TagVal v)
{
    auto o = pytype_genericalloc(RTVal_Type);
    auto self = (_RuntimeValue*)o;
    self->datatype = v.type;
    // self->cache_err = EvalError::NoError;
    self->cache_val = v.val;
    self->type_ = Const;
    self->age = (unsigned)-1;
    self->arg0 = (_RuntimeValue*)py_newref(Py_None);
    self->arg1 = (_RuntimeValue*)py_newref(Py_None);
    self->cb_arg2 = py_newref(Py_None);
    return self;
}

__attribute__((returns_nonnull,visibility("protected"))) PyObject*
new_expr2_wrap1(ValueType type, PyObject *arg0, PyObject *arg1)
{
    py_object rtarg0;
    py_object rtarg1;
    if (!is_rtval(arg0)) {
        rtarg0.reset((PyObject*)_new_const(TagVal::from_py(arg0)));
        rtarg1.reset(py_newref(arg1));
    }
    else {
        if (is_rtval(arg1)) {
            rtarg1.reset(py_newref(arg1));
        }
        else {
            rtarg1.reset((PyObject*)_new_const(TagVal::from_py(arg1)));
        }
        rtarg0.reset(py_newref(arg0));
    }
    auto datatype = binary_return_type(type, ((_RuntimeValue*)rtarg0.get())->datatype,
                                       ((_RuntimeValue*)rtarg1.get())->datatype);
    auto o = pytype_genericalloc(RTVal_Type);
    auto self = (_RuntimeValue*)o;
    self->datatype = datatype;
    // self->cache_err = EvalError::NoError;
    // self->cache_val = { .i64_val = 0 };
    self->type_ = type;
    self->age = (unsigned)-1;
    self->arg0 = (_RuntimeValue*)rtarg0.release();
    self->arg1 = (_RuntimeValue*)rtarg1.release();
    self->cb_arg2 = py_newref(Py_None);
    return o;
}

static inline __attribute__((returns_nonnull)) _RuntimeValue*
_wrap_rtval(PyObject *v)
{
    if (is_rtval(v))
        return (_RuntimeValue*)py_newref(v);
    return _new_const(TagVal::from_py(v));
}

__attribute__((returns_nonnull,visibility("protected"))) _RuntimeValue*
_new_select(_RuntimeValue *arg0, PyObject *arg1, PyObject *arg2)
{
    py_object rtarg1((PyObject*)_wrap_rtval(arg1));
    py_object rtarg2((PyObject*)_wrap_rtval(arg2));
    auto datatype = promote_type(((_RuntimeValue*)rtarg1.get())->datatype,
                                 ((_RuntimeValue*)rtarg2.get())->datatype);
    auto o = pytype_genericalloc(RTVal_Type);
    auto self = (_RuntimeValue*)o;
    self->datatype = datatype;
    // self->cache_err = EvalError::NoError;
    // self->cache_val = { .i64_val = 0 };
    self->type_ = Select;
    self->age = (unsigned)-1;
    self->arg0 = py_newref(arg0);
    self->arg1 = (_RuntimeValue*)rtarg1.release();
    self->cb_arg2 = rtarg2.release();
    return self;
}

__attribute__((visibility("hidden")))
void init()
{
    _import_array();
}

__attribute__((flatten,visibility("protected")))
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
        py_throw_format(PyExc_ValueError, "Cannot evaluate unknown argument");
    case Const:
        return;
    case Extern:
        set_cache_py(_PyObject_Vectorcall(self->cb_arg2, nullptr, 0, nullptr));
        return;
    case ExternAge: {
        if (!pyage)
            pyage.reset(pylong_from_long(age));
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
        py_throw_format(PyExc_ValueError, "Unknown value type");
    }
}

static inline bool is_numpy_int(PyObject *value)
{
    if (PyArray_IsScalar(value, Integer))
        return true;
    return PyArray_IsZeroDim(value) && PyArray_ISINTEGER((PyArrayObject*)value);
}

__attribute__((visibility("protected")))
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
static inline int get_label_offset(ValueType op, DataType t1, DataType t2)
{
    return interp_label_offsets[get_label_id(op, t1, t2)];
}

__attribute__((visibility("protected")))
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
            throw0();
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
        py_throw_format(PyExc_ValueError, "Unknown value type");
    }
}

__attribute__((visibility("protected")))
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
