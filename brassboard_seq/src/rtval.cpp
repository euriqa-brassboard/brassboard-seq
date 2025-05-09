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

#include "rtval.h"

#include <array>

#include <numpy/arrayobject.h>

namespace brassboard_seq::rtval {

__attribute__((visibility("protected"))) rtval_ref
new_cb_arg2(ValueType type, py::ptr<> cb_arg2, py::ptr<> ty)
{
    return RuntimeValue::alloc(type, pytype_to_datatype(ty),
                               py::new_none(), py::new_none(), cb_arg2);
}

__attribute__((visibility("protected"))) rtval_ref
new_expr1(ValueType type, rtval_ref &&arg0)
{
    return RuntimeValue::alloc(type, unary_return_type(type, arg0->datatype),
                               std::move(arg0), py::new_none(), py::new_none());
}

__attribute__((visibility("protected"))) rtval_ref
new_expr2(ValueType type, rtval_ref &&arg0, rtval_ref &&arg1)
{
    return RuntimeValue::alloc(type,
                               binary_return_type(type, arg0->datatype, arg1->datatype),
                               std::move(arg0), std::move(arg1), py::new_none());
}

__attribute__((visibility("protected"))) rtval_ref
new_const(TagVal v)
{
    auto self = RuntimeValue::alloc(Const, v.type, py::new_none(),
                                    py::new_none(), py::new_none());
    self->cache_val = v.val;
    return self;
}

static rtval_ref new_expr2_wrap1(ValueType type, py::ptr<> arg0, py::ptr<> arg1)
{
    rtval_ref rtarg0;
    rtval_ref rtarg1;
    if (!is_rtval(arg0)) {
        rtarg0.take(new_const(TagVal::from_py(arg0)));
        rtarg1.assign(arg1);
    }
    else {
        if (is_rtval(arg1)) {
            rtarg1.assign(arg1);
        }
        else {
            rtarg1.take(new_const(TagVal::from_py(arg1)));
        }
        rtarg0.assign(arg0);
    }
    return RuntimeValue::alloc(type, binary_return_type(type, rtarg0->datatype,
                                                        rtarg1->datatype),
                               std::move(rtarg0), std::move(rtarg1), py::new_none());
}

static inline rtval_ref wrap_rtval(py::ptr<> v)
{
    if (is_rtval(v))
        return v.ref();
    return new_const(TagVal::from_py(v));
}

__attribute__((visibility("protected")))
rtval_ref new_select(rtval_ptr arg0, py::ptr<> arg1, py::ptr<> arg2)
{
    auto rtarg1 = wrap_rtval(arg1);
    auto rtarg2 = wrap_rtval(arg2);
    return RuntimeValue::alloc(Select, promote_type(rtarg1->datatype, rtarg2->datatype),
                               arg0, std::move(rtarg1), std::move(rtarg2));
}

static inline bool tagval_equal(const TagVal &v1, const TagVal &v2)
{
    return !CmpEQ_op::generic_eval(v1, v2).is_zero();
}

static inline bool _rt_equal_val(rtval_ptr rtval, py::ptr<> pyval)
{
    if (rtval->type_ != Const)
        return false;
    return tagval_equal(rtval_cache(rtval), TagVal::from_py(pyval));
}

static bool _rt_same_value(rtval_ptr v1, rtval_ptr v2)
{
    if (v1 == v2)
        return true;
    if (v1->type_ != v2->type_)
        return false;
    switch (v1->type_) {
    case Arg: {
        auto i1 = v1->cb_arg2.as_int();
        auto i2 = v2->cb_arg2.as_int();
        if (i1 < 0 || i2 < 0) [[unlikely]]
            return false;
        return i1 == i2;
    }
    case Const:
        return tagval_equal(rtval_cache(v1), rtval_cache(v2));
    default:
    case Extern:
    case ExternAge:
        return false;
    case Not:
    case Bool:
    case Abs:
    case Ceil:
    case Floor:
    case Exp:
    case Expm1:
    case Log:
    case Log1p:
    case Log2:
    case Log10:
    case Sqrt:
    case Asin:
    case Acos:
    case Atan:
    case Asinh:
    case Acosh:
    case Atanh:
    case Sin:
    case Cos:
    case Tan:
    case Sinh:
    case Cosh:
    case Tanh:
    case Rint:
    case Int64:
        return _rt_same_value(v1->arg0, v2->arg0);
    case Select:
        return (_rt_same_value(v1->arg0, v2->arg0) && _rt_same_value(v1->arg1, v2->arg1) &&
                _rt_same_value(v1->cb_arg2, v2->cb_arg2));
    case Sub:
    case Div:
    case Pow:
    case Mod:
    case CmpLT:
    case CmpGT:
    case CmpLE:
    case CmpGE:
    case Atan2:
        return (_rt_same_value(v1->arg0, v2->arg0) && _rt_same_value(v1->arg1, v2->arg1));
    case Add:
    case Mul:
    case And:
    case Or:
    case Xor:
    case CmpNE:
    case CmpEQ:
    case Hypot:
    case Max:
    case Min:
        return ((_rt_same_value(v1->arg0, v2->arg0) && _rt_same_value(v1->arg1, v2->arg1)) ||
                (_rt_same_value(v1->arg0, v2->arg1) && _rt_same_value(v1->arg1, v2->arg0)));
    }
}

__attribute__((visibility("protected")))
bool same_value(py::ptr<> v1, py::ptr<> v2) try {
    if (!is_rtval(v1)) {
        if (!is_rtval(v2)) {
            int res = PyObject_RichCompareBool(v1, v2, Py_EQ);
            throw_if(res < 0);
            return res;
        }
        return _rt_equal_val(v2, v1);
    }
    else if (!is_rtval(v2)) {
        return _rt_equal_val(v1, v2);
    }
    return _rt_same_value(v1, v2);
}
catch (...) {
    PyErr_Clear();
    return false;
}

__attribute__((visibility("hidden")))
void init()
{
    _import_array();
    throw_if(PyType_Ready(&RuntimeValue::Type) < 0);
    throw_if(PyType_Ready(&ExternCallback::Type) < 0);
}

using extern_cb_t = TagVal(ExternCallback*);
using extern_age_cb_t = TagVal(ExternCallback*, unsigned);

__attribute__((flatten,visibility("protected")))
void rt_eval_cache(rtval_ptr self, unsigned age)
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

    auto type = self->type_;
    switch (type) {
    case Arg:
        py_throw_format(PyExc_ValueError, "Cannot evaluate unknown argument");
    case Const:
        return;
    case Extern:
    case ExternAge: {
        auto ecb = (ExternCallback*)self->cb_arg2;
        auto val = (type == Extern ? ((extern_cb_t*)ecb->fptr)(ecb) :
                    ((extern_age_cb_t*)ecb->fptr)(ecb, age));
        set_cache(val.convert(self->datatype));
        return;
    }
    default:
        break;
    }

    rtval_ptr rtarg0 = self->arg0;
    rt_eval_cache(rtarg0, age);
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

    rtval_ptr rtarg1 = self->arg1;
    if (type == Select) {
        rtval_ptr rtarg2 = self->cb_arg2;
        auto rtres = arg0.get<bool>() ? rtarg1 : rtarg2;
        rt_eval_cache(rtres, age);
        set_cache(rtval_cache(rtres).convert(self->datatype));
        return;
    }
    rt_eval_cache(rtarg1, age);
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
TagVal TagVal::from_py(py::ptr<> value)
{
    if (value == Py_True)
        return true;
    if (value == Py_False)
        return false;
    if (value.isa<py::int_>() || is_numpy_int(value))
        return TagVal(value.as_int<int64_t>());
    return TagVal(value.as_float());
}

static inline TagVal tagval_add_or_sub(TagVal v1, TagVal v2, bool issub)
{
    assert(v1.err == EvalError::NoError);
    assert(v2.err == EvalError::NoError);
    return (issub ? Sub_op::generic_eval(v1, v2) : Add_op::generic_eval(v1, v2));
}

static inline rtval_ref new_addsub(TagVal c, rtval_ptr v, bool s)
{
    if (c.is_zero() && !s)
        return v.ref();
    return new_expr2(s ? Sub : Add, new_const(c), v);
}

static inline rtval_ref build_addsub(py::ptr<> v0, py::ptr<> v1, bool issub)
{
    assume(v0);
    assume(v1);
    TagVal nc{};
    rtval_ptr nv0;
    bool ns0 = false;
    if (!is_rtval(v0)) {
        nc = TagVal::from_py(v0);
        assert(is_rtval(v1));
        assume(is_rtval(v1));
    }
    else {
        nv0 = (RuntimeValue*)v0;
        auto type = nv0->type_;
        if (type == Const) {
            nc = rtval_cache(nv0);
            nv0 = nullptr;
        }
        else if (type == Add) {
            rtval_ptr arg0 = nv0->arg0;
            // Add/Sub should only have the first argument as constant
            if (arg0->type_ == Const) {
                nc = rtval_cache(arg0);
                nv0 = nv0->arg1;
            }
        }
        else if (type == Sub) {
            rtval_ptr arg0 = nv0->arg0;
            // Add/Sub should only have the first argument as constant
            if (arg0->type_ == Const) {
                ns0 = true;
                nc = rtval_cache(arg0);
                nv0 = nv0->arg1;
            }
        }
    }
    bool ns1 = false;
    rtval_ptr nv1;
    if (!is_rtval(v1)) {
        nc = tagval_add_or_sub(nc, TagVal::from_py(v1), issub);
    }
    else {
        nv1 = (RuntimeValue*)v1;
        auto type = nv1->type_;
        if (type == Const) {
            nc = tagval_add_or_sub(nc, rtval_cache(nv1), issub);
            nv1 = nullptr;
        }
        else if (type == Add) {
            rtval_ptr arg0 = nv1->arg0;
            // Add/Sub should only have the first argument as constant
            if (arg0->type_ == Const) {
                nc = tagval_add_or_sub(nc, rtval_cache(arg0), issub);
                nv1 = nv1->arg1;
            }
        }
        else if (type == Sub) {
            rtval_ptr arg0 = nv1->arg0;
            // Add/Sub should only have the first argument as constant
            if (arg0->type_ == Const) {
                ns1 = true;
                nc = tagval_add_or_sub(nc, rtval_cache(arg0), issub);
                nv1 = nv1->arg1;
            }
        }
    }
    if (nv0 == v0 && nv1 == v1)
        return new_expr2(issub ? Sub : Add, nv0, nv1);
    if (issub)
        ns1 = !ns1;
    if (!nv0) {
        if (!nv1)
            return new_const(nc);
        return new_addsub(nc, nv1, ns1);
    }
    if (!nv1)
        return new_addsub(nc, nv0, ns0);
    bool ns = false;
    rtval_ref nv;
    if (!ns0) {
        nv.take(new_expr2(ns1 ? Sub : Add, nv0, nv1));
    }
    else if (ns1) {
        nv.take(new_expr2(Add, nv0, nv1));
        ns = true;
    }
    else {
        nv.take(new_expr2(Sub, nv1, nv0));
    }
    return new_addsub(nc, nv, ns);
}

namespace np {
#define GET_NP(name) static auto const name = "numpy"_pymod.attr(#name).rel()
GET_NP(add);
GET_NP(subtract);
GET_NP(multiply);
GET_NP(divide);
GET_NP(remainder);
GET_NP(bitwise_and);
GET_NP(bitwise_or);
GET_NP(bitwise_xor);
GET_NP(logical_not);
GET_NP(power);
GET_NP(less);
GET_NP(greater);
GET_NP(less_equal);
GET_NP(greater_equal);
GET_NP(equal);
GET_NP(not_equal);

GET_NP(fmin);
GET_NP(fmax);

GET_NP(abs);
GET_NP(ceil);
GET_NP(exp);
GET_NP(expm1);
GET_NP(floor);
GET_NP(log);
GET_NP(log1p);
GET_NP(log2);
GET_NP(log10);
GET_NP(sqrt);
GET_NP(arcsin);
GET_NP(arccos);
GET_NP(arctan);
GET_NP(arctan2);
GET_NP(arcsinh);
GET_NP(arccosh);
GET_NP(arctanh);
GET_NP(sin);
GET_NP(cos);
GET_NP(tan);
GET_NP(sinh);
GET_NP(cosh);
GET_NP(tanh);
GET_NP(hypot);
GET_NP(rint);
#undef GET_NP
};

static inline bool is_integer(auto v)
{
    return (v->datatype != DataType::Float64 || v->type_ == Ceil ||
            v->type_ == Floor || v->type_ == Rint);
}

static py::ref<> rtvalue_array_ufunc(rtval_ptr self, PyObject *const *args,
                                     Py_ssize_t nargs)
{
    py::check_num_arg("RuntimeValue.__array_ufunc__", nargs, 2);
    auto ufunc = args[0];
    if (py::str(args[1]).compare_ascii("__call__") != 0)
        return py::new_not_implemented();
    // numpy type support would dispatch arithmetic operations to this function
    // so we need to implement the corresponding ufuncs to support these.
    if (ufunc == np::add) {
        py::check_num_arg("RuntimeValue.__array_ufunc__", nargs, 4, 4);
        return build_addsub(args[2], args[3], false);
    }
    if (ufunc == np::subtract) {
        py::check_num_arg("RuntimeValue.__array_ufunc__", nargs, 4, 4);
        return build_addsub(args[2], args[3], true);
    }
    auto uni_expr = [&] (auto type) { return new_expr1(type, self); };
    auto bin_expr = [&] (auto type) {
        py::check_num_arg("RuntimeValue.__array_ufunc__", nargs, 4, 4);
        return new_expr2_wrap1(type, args[2], args[3]);
    };
    if (ufunc == np::multiply)
        return bin_expr(Mul);
    if (ufunc == np::divide)
        return bin_expr(Div);
    if (ufunc == np::remainder)
        return bin_expr(Mod);
    if (ufunc == np::bitwise_and)
        return bin_expr(And);
    if (ufunc == np::bitwise_or)
        return bin_expr(Or);
    if (ufunc == np::bitwise_xor)
        return bin_expr(Xor);
    if (ufunc == np::logical_not)
        return (self->type_ == Not ? rt_convert_bool(self->arg0) : uni_expr(Not));
    if (ufunc == np::power)
        return bin_expr(Pow);
    if (ufunc == np::less)
        return bin_expr(CmpLT);
    if (ufunc == np::greater)
        return bin_expr(CmpGT);
    if (ufunc == np::less_equal)
        return bin_expr(CmpLE);
    if (ufunc == np::greater_equal)
        return bin_expr(CmpGE);
    if (ufunc == np::equal)
        return bin_expr(CmpEQ);
    if (ufunc == np::not_equal)
        return bin_expr(CmpNE);
    if (ufunc == np::fmin) {
        py::check_num_arg("RuntimeValue.__array_ufunc__", nargs, 4, 4);
        return args[2] == args[3] ? self.ref() : bin_expr(Min);
    }
    if (ufunc == np::fmax) {
        py::check_num_arg("RuntimeValue.__array_ufunc__", nargs, 4, 4);
        return args[2] == args[3] ? self.ref() : bin_expr(Max);
    }
    if (ufunc == np::abs)
        return self->type_ == Abs ? self.ref() : uni_expr(Abs);
    if (ufunc == np::ceil)
        return is_integer(self) ? self.ref() : uni_expr(Ceil);
    if (ufunc == np::exp)
        return uni_expr(Exp);
    if (ufunc == np::expm1)
        return uni_expr(Expm1);
    if (ufunc == np::floor)
        return is_integer(self) ? self.ref() : uni_expr(Floor);
    if (ufunc == np::log)
        return uni_expr(Log);
    if (ufunc == np::log1p)
        return uni_expr(Log1p);
    if (ufunc == np::log2)
        return uni_expr(Log2);
    if (ufunc == np::log10)
        return uni_expr(Log10);
    if (ufunc == np::sqrt)
        return uni_expr(Sqrt);
    if (ufunc == np::arcsin)
        return uni_expr(Asin);
    if (ufunc == np::arccos)
        return uni_expr(Acos);
    if (ufunc == np::arctan)
        return uni_expr(Atan);
    if (ufunc == np::arctan2)
        return bin_expr(Atan2);
    if (ufunc == np::arcsinh)
        return uni_expr(Asinh);
    if (ufunc == np::arccosh)
        return uni_expr(Acosh);
    if (ufunc == np::arctanh)
        return uni_expr(Atanh);
    if (ufunc == np::sin)
        return uni_expr(Sin);
    if (ufunc == np::cos)
        return uni_expr(Cos);
    if (ufunc == np::tan)
        return uni_expr(Tan);
    if (ufunc == np::sinh)
        return uni_expr(Sinh);
    if (ufunc == np::cosh)
        return uni_expr(Cosh);
    if (ufunc == np::tanh)
        return uni_expr(Tanh);
    if (ufunc == np::hypot)
        return bin_expr(Hypot);
    if (ufunc == np::rint)
        return is_integer(self) ? self.ref() : uni_expr(Rint);
    return py::new_not_implemented();
}

static auto rtvalue_eval(rtval_ptr self, py::ptr<> pyage)
{
    rt_eval_cache(self, pyage.as_int());
    return rtval_cache(self).to_py();
}

static auto rtvalue_ceil(rtval_ptr self)
{
    return is_integer(self) ? self.ref() : new_expr1(Ceil, self);
}

static auto rtvalue_floor(rtval_ptr self)
{
    return is_integer(self) ? self.ref() : new_expr1(Floor, self);
}

static auto rtvalue_round(rtval_ptr self)
{
    return rt_round_int64(self);
}

static inline constexpr int operator_precedence(ValueType type_)
{
    if (type_ == Add || type_ == Sub)
        return 3;
    if (type_ == Mul || type_ == Div)
        return 2;
    if (type_ == CmpLT || type_ == CmpGT || type_ == CmpLE || type_ == CmpGE)
        return 4;
    if (type_ == CmpNE || type_ == CmpEQ)
        return 5;
    if (type_ == And)
        return 6;
    if (type_ == Or)
        return 8;
    if (type_ == Pow)
        return 7;
    return 0;
}

static inline bool needs_parenthesis(auto v, ValueType parent_type)
{
    auto op_self = operator_precedence(v->type_);
    auto op_parent = operator_precedence(parent_type);
    if (op_self == 0 || op_parent == 0 || op_self < op_parent)
        return false;
    if (op_self > op_parent)
        return true;
    return !(parent_type == Add || parent_type == Mul);
}

namespace {

struct rtvalue_printer : py::stringio {
    void show_arg(rtval_ptr v, ValueType parent_type)
    {
        auto p = needs_parenthesis(v, parent_type);
        if (p) write_ascii("(");
        show(v);
        if (p) write_ascii(")");
    }
    void show_binary(rtval_ptr v, const char *op, ValueType type_)
    {
        show_arg(v->arg0, type_);
        write_ascii(op);
        show_arg(v->arg1, type_);
    }
    void show_call1(rtval_ptr v, const char *f)
    {
        write_ascii(f);
        write_ascii("(");
        show(v->arg0);
        write_ascii(")");
    }
    void show_call2(rtval_ptr v, const char *f)
    {
        write_ascii(f);
        write_ascii("(");
        show(v->arg0);
        write_ascii(", ");
        show(v->arg1);
        write_ascii(")");
    }
    void show_call3(rtval_ptr v, const char *f)
    {
        write_ascii(f);
        write_ascii("(");
        show(v->arg0);
        write_ascii(", ");
        show(v->arg1);
        write_ascii(", ");
        show((RuntimeValue*)v->cb_arg2);
        write_ascii(")");
    }
    void show(rtval_ptr v)
    {
        switch (v->type_) {
        case Extern:
        case ExternAge:
            return write_str(v->cb_arg2);
        case Arg:
            write_ascii("arg(");
            write_str(v->cb_arg2);
            return write_ascii(")");
        case Const:
            return write_str(rtval_cache(v).to_py());
        case Add:
            return show_binary(v, " + ", v->type_);
        case Sub:
            return show_binary(v, " - ", v->type_);
        case Mul:
            return show_binary(v, " * ", v->type_);
        case Div:
            return show_binary(v, " / ", v->type_);
        case CmpLT:
            return show_binary(v, " < ", v->type_);
        case CmpGT:
            return show_binary(v, " > ", v->type_);
        case CmpLE:
            return show_binary(v, " <= ", v->type_);
        case CmpGE:
            return show_binary(v, " >= ", v->type_);
        case CmpEQ:
            return show_binary(v, " == ", v->type_);
        case CmpNE:
            return show_binary(v, " != ", v->type_);
        case And:
            return show_binary(v, " & ", v->type_);
        case Or:
            return show_binary(v, " | ", v->type_);
        case Xor:
            return show_binary(v, " ^ ", v->type_);
        case Mod:
            return show_binary(v, " % ", v->type_);
        case Pow:
            return show_binary(v, "**", v->type_);
        case Not:
            return show_call1(v, "inv");
        case Abs:
            return show_call1(v, "abs");
        case Ceil:
            return show_call1(v, "ceil");
        case Exp:
            return show_call1(v, "exp");
        case Expm1:
            return show_call1(v, "expm1");
        case Floor:
            return show_call1(v, "floor");
        case Log:
            return show_call1(v, "log");
        case Log1p:
            return show_call1(v, "log1p");
        case Log2:
            return show_call1(v, "log2");
        case Log10:
            return show_call1(v, "log10");
        case Sqrt:
            return show_call1(v, "sqrt");
        case Asin:
            return show_call1(v, "arcsin");
        case Acos:
            return show_call1(v, "arccos");
        case Atan:
            return show_call1(v, "arctan");
        case Asinh:
            return show_call1(v, "arcsinh");
        case Acosh:
            return show_call1(v, "arccosh");
        case Atanh:
            return show_call1(v, "arctanh");
        case Sin:
            return show_call1(v, "sin");
        case Cos:
            return show_call1(v, "cos");
        case Tan:
            return show_call1(v, "tan");
        case Sinh:
            return show_call1(v, "sinh");
        case Cosh:
            return show_call1(v, "cosh");
        case Tanh:
            return show_call1(v, "tanh");
        case Rint:
            return show_call1(v, "rint");
        case Max:
            return show_call2(v, "max");
        case Min:
            return show_call2(v, "min");
        case Int64:
            return show_call1(v, "int64");
        case Bool:
            return show_call1(v, "bool");
        case Atan2:
            return show_call2(v, "arctan2");
        case Hypot:
            return show_call2(v, "hypot");
        case Select:
            return show_call3(v, "ifelse");
        default:
            return write_ascii("Unknown value");
        }
    }
};

}

static constexpr auto rtvalue_str = py::unifunc<[] (rtval_ptr self) {
    rtvalue_printer io;
    io.show(self);
    return io.getvalue();
}>;

static auto rtvalue_as_number = PyNumberMethods{
    .nb_add = py::binfunc<[] (auto v1, auto v2) {
        return build_addsub(v1, v2, false); }>,
    .nb_subtract = py::binfunc<[] (auto v1, auto v2) {
        return build_addsub(v1, v2, true); }>,
    .nb_multiply = py::binfunc<[] (auto v1, auto v2) {
        return new_expr2_wrap1(Mul, v1, v2); }>,
    .nb_remainder = py::binfunc<[] (auto v1, auto v2) {
        return new_expr2_wrap1(Mod, v1, v2); }>,
    .nb_power = py::trifunc<[] (auto v1, auto v2, auto v3) -> py::ref<> {
        if (v3 != Py_None) [[unlikely]]
            return py::new_not_implemented();
        return new_expr2_wrap1(Pow, v1, v2);
    }>,
    .nb_negative = py::unifunc<[] (auto self) {
        return build_addsub(py::int_cached(0), self, true);
    }>,
    .nb_positive = [] (PyObject *self) { return py::newref(self); },
    .nb_absolute = py::unifunc<[] (rtval_ptr self) {
        if (self->type_ == Abs)
            return self.ref();
        return new_expr1(Abs, self);
    }>,
    .nb_bool = [] (auto) {
        // It's too easy to accidentally use this in control flow/assertion
        PyErr_Format(PyExc_TypeError, "Cannot convert runtime value to boolean");
        return -1;
    },
    .nb_and = py::binfunc<[] (auto v1, auto v2) {
        return new_expr2_wrap1(And, v1, v2); }>,
    .nb_xor = py::binfunc<[] (auto v1, auto v2) {
        return new_expr2_wrap1(Xor, v1, v2); }>,
    .nb_or = py::binfunc<[] (auto v1, auto v2) {
        return new_expr2_wrap1(Or, v1, v2); }>,
    .nb_true_divide = py::binfunc<[] (auto v1, auto v2) {
        return new_expr2_wrap1(Div, v1, v2); }>,
};
__attribute__((visibility("protected")))
PyTypeObject RuntimeValue::Type = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "brassboard_seq.rtval.RuntimeValue",
    .tp_basicsize = sizeof(RuntimeValue),
    .tp_dealloc = py::tp_cxx_dealloc<true,RuntimeValue>,
    .tp_repr = rtvalue_str,
    .tp_as_number = &rtvalue_as_number,
    .tp_str = rtvalue_str,
    .tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_HAVE_GC,
    .tp_traverse = py::tp_field_traverse<RuntimeValue,&RuntimeValue::arg0,
    &RuntimeValue::arg1,&RuntimeValue::cb_arg2>,
    .tp_clear = py::tp_field_clear<RuntimeValue,&RuntimeValue::arg0,
    &RuntimeValue::arg1,&RuntimeValue::cb_arg2>,
    .tp_richcompare = py::tp_richcompare<[] (auto v1, auto v2, int op) -> py::ref<> {
        auto typ = pycmp2valcmp(op);
        if (is_rtval(v2)) {
            if (v1 == v2)
                return py::new_bool(typ == CmpLE || typ == CmpGE || typ == CmpEQ);
            return new_expr2(typ, v1, v2);
        }
        return new_expr2(typ, v1, new_const(TagVal::from_py(v2)));
    }>,
    .tp_methods = (py::meth_table<
                   py::meth_fast<"__array_ufunc__",rtvalue_array_ufunc>,
                   py::meth_o<"eval",rtvalue_eval>,
                   py::meth_noargs<"__ceil__",rtvalue_ceil>,
                   py::meth_noargs<"__floor__",rtvalue_floor>,
                   py::meth_noargs<"__round__",rtvalue_round>>),
};

__attribute__((visibility("protected")))
PyTypeObject ExternCallback::Type = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "brassboard_seq.rtval.ExternCallback",
    .tp_basicsize = sizeof(ExternCallback),
    .tp_dealloc = py::tp_cxx_dealloc<false,ExternCallback>,
    .tp_flags = Py_TPFLAGS_DEFAULT,
};

PyMethodDef methods[] = {
    py::meth_fast<"get_value",[] (auto, PyObject *const *args, Py_ssize_t nargs) {
        py::check_num_arg("get_value", nargs, 2, 2);
        auto pyage = py::arg_cast<py::int_>(args[1], "age");
        if (is_rtval(args[0])) {
            rt_eval_cache(args[0], pyage.as_int());
            return rtval_cache(args[0]).to_py();
        }
        return py::ptr(args[0]).ref();
    }>,
    py::meth_o<"inv",[] (auto, py::ptr<> v) -> py::ref<> {
        if (v == Py_True)
            return py::new_false();
        if (v == Py_False)
            return py::new_true();
        if (auto rv = py::cast<RuntimeValue>(v)) {
            if (rv->type_ == Not)
                return rt_convert_bool(rv->arg0);
            return new_expr1(Not, rv);
        }
        return py::new_bool(!v.as_bool());
    }>,
    py::meth_o<"convert_bool",[] (auto, py::ptr<> v) -> py::ref<> {
        if (auto rv = py::cast<RuntimeValue>(v))
            return rt_convert_bool(rv);
        return py::new_bool(v.as_bool());
    }>,
    py::meth_fast<"ifelse",[] (auto, PyObject *const *args, Py_ssize_t nargs) -> py::ref<> {
        py::check_num_arg("ifelse", nargs, 3, 3);
        auto b = py::ptr(args[0]);
        auto v1 = py::ptr(args[1]);
        auto v2 = py::ptr(args[2]);
        if (same_value(v1, v2))
            return v1.ref();
        if (auto rb = py::cast<RuntimeValue>(b))
            return new_select(rb, v1, v2);
        return (b.as_bool() ? v1 : v2).ref();
    }>,
    py::meth_fast<"same_value",[] (auto, PyObject *const *args, Py_ssize_t nargs) {
        py::check_num_arg("same_value", nargs, 2, 2);
        return py::new_bool(same_value(args[0], args[1]));
    }>, {}};

static __attribute__((flatten, noinline))
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
    auto eval_uni = [&] <typename op,DataType t1> {
        auto [eo, out] = pop_operand();
        auto [e1, in1] = pop_operand();
        using T1 = data_type_t<t1>;
        using Tout = data_type_t<op::return_type(t1)>;
        auto res = op::template eval_err<Tout,T1>(in1->get<T1>());
        *eo = combine_error(*e1, res.err);
        *out = res.val;
    };
    auto eval_bin = [&] <typename op,DataType t1,DataType t2> {
        auto [eo, out] = pop_operand();
        auto [e1, in1] = pop_operand();
        auto [e2, in2] = pop_operand();
        using T1 = data_type_t<t1>;
        using T2 = data_type_t<t2>;
        using Tout = data_type_t<op::return_type(t1, t2)>;
        auto res = op::template eval_err<Tout,T1,T2>(in1->get<T1>(), in2->get<T2>());
        *eo = combine_error(*e1, *e2, res.err);
        *out = res.val;
    };
    auto eval_select = [&] <DataType t1,DataType t2> {
        auto [eo, out] = pop_operand();
        auto [e0, in0] = pop_operand();
        auto [e1, in1] = pop_operand();
        auto [e2, in2] = pop_operand();
        bool b = bool(in0->i64_val);
        using T1 = data_type_t<t1>;
        using T2 = data_type_t<t2>;
        using Tout = data_type_t<promote_type(t1, t2)>;
        out->get<Tout>() = b ? Tout(in1->get<T1>()) : Tout(in2->get<T2>());
        *eo = combine_error(*e0, b ? *e1 : *e2);
    };

    goto *pop_label();

return_label: {
        auto [eo, out] = pop_operand();
        return {*eo, *out};
    }

#define GEN_UNI_OP(f, t1) f##_op_##t1##_label:                  \
    eval_uni.template operator()<f##_op,DataType::t1>();        \
    goto *pop_label();
#define GEN_BIN_OP(f, t1, t2) f##_op_##t1##_##t2##_label:               \
    eval_bin.template operator()<f##_op,DataType::t1,DataType::t2>();   \
    goto *pop_label();
#define GEN_SELECT_OP(t1, t2) Select_op_##t1##_##t2##_label:            \
    eval_select.template operator()<DataType::t1,DataType::t2>();       \
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
#define GEN_UNI_OP(f, t1) res[get_label_id(f, DataType::t1)] = label_offsets[idx++];
#define GEN_BIN_OP(f, t1, t2)                                           \
    res[get_label_id(f, DataType::t1, DataType::t2)] = label_offsets[idx++];
#define GEN_SELECT_OP(t2, t3)                                           \
    res[get_label_id(Select, DataType::t2, DataType::t3)] = label_offsets[idx++];
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
void InterpFunction::set_value(rtval_ptr value, std::vector<DataType> &args)
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
        auto v = value->cb_arg2.as_int();
        if (v < 0 || v >= builder.nargs)
            py_throw_format(PyExc_IndexError, "Argument index out of bound: %ld.", v);
        info.val.type = builder.types[v];
        info.dynamic = true;
        info.inited = true;
        info.idx = v;
        return info;
    }
    case Extern:
    case ExternAge: {
        info.val.type = value->datatype;
        info.inited = true;
        return info;
    }
    default:
        break;
    }

    rtval_ptr rtarg0 = value->arg0;
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

    switch (type) {
#define HANDLE_UNARY(op)                                                \
        case op: return handle_unary(op##_op::return_type(arg0_info.val.type))
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

    rtval_ptr rtarg1 = value->arg1;
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

    switch (type) {
#define HANDLE_BINARY(op)                                               \
        case op: return handle_binary(op##_op::return_type(arg0_info.val.type, \
                                                           arg1_info.val.type))
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
#undef HANDL_BINARY
    default:
        py_throw_format(PyExc_ValueError, "Unknown value type");
    }
}

__attribute__((visibility("protected")))
void InterpFunction::eval_all(unsigned age)
{
    for (size_t i = 0; i < rt_vals.size(); i++) {
        auto rt_val = (RuntimeValue*)rt_vals[i];
        if (!rt_val) {
            errors[i] = EvalError::NoError;
            continue;
        }
        rt_eval_cache(rt_val, age);
        data[i] = rt_val->cache_val;
        errors[i] = rt_val->cache_err;
    }
}

__attribute__((visibility("protected")))
TagVal InterpFunction::call()
{
    auto [err, val] = interpret_func(code.data(), data.data(), errors.data());
    TagVal res;
    res.type = ret_type;
    res.err = err;
    res.val = val;
    return res;
}

}
