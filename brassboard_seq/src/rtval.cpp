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

namespace brassboard_seq::rtval {

static inline TagVal tagval_add_or_sub(TagVal v1, TagVal v2, bool issub)
{
    assert(v1.err == EvalError::NoError);
    assert(v2.err == EvalError::NoError);
    return (issub ? Sub_op::generic_eval(v1, v2) : Add_op::generic_eval(v1, v2));
}

static inline __attribute__((returns_nonnull)) PyObject*
_new_addsub(TagVal c, _RuntimeValue *v, bool s)
{
    if (c.is_zero() && !s)
        return py_newref((PyObject*)v);
    py_object arg0((PyObject*)_new_const(c));
    return (PyObject*)_new_expr2(s ? Sub : Add, (_RuntimeValue*)arg0.get(), v);
}

static inline __attribute__((returns_nonnull)) PyObject*
build_addsub(PyObject *v0, PyObject *v1, bool issub)
{
    assume(v0);
    assume(v1);
    TagVal nc{};
    _RuntimeValue *nv0 = nullptr;
    bool ns0 = false;
    if (!is_rtval(v0)) {
        nc = TagVal::from_py(v0);
        assert(is_rtval(v1));
        assume(is_rtval(v1));
    }
    else {
        nv0 = (_RuntimeValue*)v0;
        auto type = nv0->type_;
        if (type == Const) {
            nc = rtval_cache(nv0);
            nv0 = nullptr;
        }
        else if (type == Add) {
            auto arg0 = nv0->arg0;
            // Add/Sub should only have the first argument as constant
            if (arg0->type_ == Const) {
                nc = rtval_cache(arg0);
                nv0 = nv0->arg1;
            }
        }
        else if (type == Sub) {
            auto arg0 = nv0->arg0;
            // Add/Sub should only have the first argument as constant
            if (arg0->type_ == Const) {
                ns0 = true;
                nc = rtval_cache(arg0);
                nv0 = nv0->arg1;
            }
        }
    }
    bool ns1 = false;
    _RuntimeValue *nv1 = nullptr;
    if (!is_rtval(v1)) {
        nc = tagval_add_or_sub(nc, TagVal::from_py(v1), issub);
    }
    else {
        nv1 = (_RuntimeValue*)v1;
        auto type = nv1->type_;
        if (type == Const) {
            nc = tagval_add_or_sub(nc, rtval_cache(nv1), issub);
            nv1 = nullptr;
        }
        else if (type == Add) {
            auto arg0 = nv1->arg0;
            // Add/Sub should only have the first argument as constant
            if (arg0->type_ == Const) {
                nc = tagval_add_or_sub(nc, rtval_cache(arg0), issub);
                nv1 = nv1->arg1;
            }
        }
        else if (type == Sub) {
            auto arg0 = nv1->arg0;
            // Add/Sub should only have the first argument as constant
            if (arg0->type_ == Const) {
                ns1 = true;
                nc = tagval_add_or_sub(nc, rtval_cache(arg0), issub);
                nv1 = nv1->arg1;
            }
        }
    }
    if ((PyObject*)nv0 == v0 && (PyObject*)nv1 == v1)
        return (PyObject*)_new_expr2(issub ? Sub : Add, nv0, nv1);
    if (issub)
        ns1 = !ns1;
    if (!nv0) {
        if (!nv1)
            return (PyObject*)_new_const(nc);
        return _new_addsub(nc, nv1, ns1);
    }
    if (!nv1)
        return _new_addsub(nc, nv0, ns0);
    bool ns = false;
    py_object nv;
    if (!ns0) {
        nv.reset((PyObject*)_new_expr2(ns1 ? Sub : Add, nv0, nv1));
    }
    else if (ns1) {
        nv.reset((PyObject*)_new_expr2(Add, nv0, nv1));
        ns = true;
    }
    else {
        nv.reset((PyObject*)_new_expr2(Sub, nv1, nv0));
    }
    return _new_addsub(nc, (_RuntimeValue*)nv.get(), ns);
}

namespace np {
static auto const mod = throw_if_not(PyImport_ImportModule("numpy"));
#define GET_NP(name)                                                    \
    static auto const name = throw_if_not(PyObject_GetAttrString(mod, #name))
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

static PyObject *rtvalue_add(PyObject *v1, PyObject *v2)
{
    return py_catch_error([&] { return build_addsub(v1, v2, false); });
}

static PyObject *rtvalue_sub(PyObject *v1, PyObject *v2)
{
    return py_catch_error([&] { return build_addsub(v1, v2, true); });
}

static PyObject *rtvalue_mul(PyObject *v1, PyObject *v2)
{
    return py_catch_error([&] { return new_expr2_wrap1(Mul, v1, v2); });
}

static PyObject *rtvalue_div(PyObject *v1, PyObject *v2)
{
    return py_catch_error([&] { return new_expr2_wrap1(Div, v1, v2); });
}

static PyObject *rtvalue_mod(PyObject *v1, PyObject *v2)
{
    return py_catch_error([&] { return new_expr2_wrap1(Mod, v1, v2); });
}

static PyObject *rtvalue_pow(PyObject *v1, PyObject *v2, PyObject *v3)
{
    if (v3 != Py_None) [[unlikely]]
        Py_RETURN_NOTIMPLEMENTED;
    return py_catch_error([&] { return new_expr2_wrap1(Pow, v1, v2); });
}

static PyObject *rtvalue_neg(PyObject *self)
{
    return py_catch_error([&] { return build_addsub(pylong_cached(0), self, true); });
}

static PyObject *rtvalue_pos(PyObject *self)
{
    return py_newref(self);
}

static PyObject *rtvalue_abs(PyObject *self)
{
    if (((_RuntimeValue*)self)->type_ == Abs)
        return py_newref(self);
    return py_catch_error([&] { return new_expr1(Abs, self); });
}

static int rtvalue_bool(PyObject*)
{
    // It's too easy to accidentally use this in control flow/assertion
    PyErr_Format(PyExc_TypeError, "Cannot convert runtime value to boolean");
    return -1;
}

static PyObject *rtvalue_and(PyObject *v1, PyObject *v2)
{
    return py_catch_error([&] { return new_expr2_wrap1(And, v1, v2); });
}

static PyObject *rtvalue_xor(PyObject *v1, PyObject *v2)
{
    return py_catch_error([&] { return new_expr2_wrap1(Xor, v1, v2); });
}

static PyObject *rtvalue_or(PyObject *v1, PyObject *v2)
{
    return py_catch_error([&] { return new_expr2_wrap1(Or, v1, v2); });
}

static PyNumberMethods rtvalue_as_number = {
    .nb_add = rtvalue_add,
    .nb_subtract = rtvalue_sub,
    .nb_multiply = rtvalue_mul,
    .nb_remainder = rtvalue_mod,
    .nb_power = rtvalue_pow,
    .nb_negative = rtvalue_neg,
    .nb_positive = rtvalue_pos,
    .nb_absolute = rtvalue_abs,
    .nb_bool = rtvalue_bool,
    .nb_and = rtvalue_and,
    .nb_xor = rtvalue_xor,
    .nb_or = rtvalue_or,
    .nb_true_divide = rtvalue_div,
};

static PyObject *rtvalue_richcmp(PyObject *v1, PyObject *v2, int op)
{
    return py_catch_error([&] {
        auto typ = pycmp2valcmp(op);
        if (is_rtval(v2)) {
            if (v1 == v2)
                return ((typ == CmpLE || typ == CmpGE || typ == CmpEQ) ?
                        py_immref(Py_True) : py_immref(Py_False));
            return new_expr2(typ, v1, v2);
        }
        py_object rv2((PyObject*)_new_const(TagVal::from_py(v2)));
        return new_expr2(typ, v1, rv2.get());
    });
}

static inline bool is_integer(auto v)
{
    return (v->datatype != DataType::Float64 || v->type_ == Ceil ||
            v->type_ == Floor || v->type_ == Rint);
}

static PyObject *rtvalue_array_ufunc(PyObject *py_self, PyObject *const *args,
                                     Py_ssize_t nargs) try
{
    py_check_num_arg("__array_ufunc__", nargs, 2);
    auto self = (_RuntimeValue*)py_self;
    auto ufunc = args[0];
    auto methods = args[1];
    if (PyUnicode_CompareWithASCIIString(methods, "__call__"))
        Py_RETURN_NOTIMPLEMENTED;
    // Needed for numpy type support
    if (ufunc == np::add)
        return build_addsub(args[2], args[3], false);
    if (ufunc == np::subtract)
        return build_addsub(args[2], args[3], true);
    auto uni_expr = [&] (auto type) { return (PyObject*)new_expr1(type, self); };
    auto bin_expr = [&] (auto type) { return new_expr2_wrap1(type, args[2], args[3]); };
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
        return (self->type_ == Not ?
                (PyObject*)rt_convert_bool(self->arg0) : uni_expr(Not));
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
    if (ufunc == np::fmin)
        return args[2] == args[3] ? py_newref(py_self) : bin_expr(Min);
    if (ufunc == np::fmax)
        return args[2] == args[3] ? py_newref(py_self) : bin_expr(Max);
    if (ufunc == np::abs)
        return self->type_ == Abs ? py_newref(py_self) : uni_expr(Abs);
    if (ufunc == np::ceil)
        return is_integer(self) ? py_newref(py_self) : uni_expr(Ceil);
    if (ufunc == np::exp)
        return uni_expr(Exp);
    if (ufunc == np::expm1)
        return uni_expr(Expm1);
    if (ufunc == np::floor)
        return is_integer(self) ? py_newref(py_self) : uni_expr(Floor);
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
        return is_integer(self) ? py_newref(py_self) : uni_expr(Rint);
    Py_RETURN_NOTIMPLEMENTED;
}
catch (...) {
    return nullptr;
}

static PyObject *rtvalue_eval(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    return py_catch_error([&] {
        py_check_num_arg("eval", nargs, 1);
        auto age = PyLong_AsLong(args[0]);
        throw_if(age == -1 && PyErr_Occurred());
        rt_eval_cache(self, age);
        return rtval_cache((_RuntimeValue*)self).to_py();
    });
}

static PyObject *rtvalue_ceil(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    return py_catch_error([&] {
        py_check_num_arg("__ceil__", nargs, 0);
        return is_integer((_RuntimeValue*)self) ? py_newref(self) :
            (PyObject*)new_expr1(Ceil, self);
    });
}

static PyObject *rtvalue_floor(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    return py_catch_error([&] {
        py_check_num_arg("__floor__", nargs, 0);
        return is_integer((_RuntimeValue*)self) ? py_newref(self) :
            (PyObject*)new_expr1(Floor, self);
    });
}

static PyObject *rtvalue_round(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    return py_catch_error([&] {
        py_check_num_arg("__round__", nargs, 0);
        return rt_round_int64((_RuntimeValue*)self);
    });
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

struct rtvalue_printer : py_stringio {
    void show_arg(_RuntimeValue *v, ValueType parent_type)
    {
        auto p = needs_parenthesis(v, parent_type);
        if (p) write("("_py);
        show(v);
        if (p) write(")"_py);
    }
    void show_binary(_RuntimeValue *v, PyObject *op, ValueType type_)
    {
        show_arg(v->arg0, type_);
        write(op);
        show_arg(v->arg1, type_);
    }
    void show_call1(_RuntimeValue *v, PyObject *f)
    {
        write(f);
        write("("_py);
        show(v->arg0);
        write(")"_py);
    }
    void show_call2(_RuntimeValue *v, PyObject *f)
    {
        write(f);
        write("("_py);
        show(v->arg0);
        write(", "_py);
        show(v->arg1);
        write(")"_py);
    }
    void show_call3(_RuntimeValue *v, PyObject *f)
    {
        write(f);
        write("("_py);
        show(v->arg0);
        write(", "_py);
        show(v->arg1);
        write(", "_py);
        show((_RuntimeValue*)v->cb_arg2);
        write(")"_py);
    }
    void write_str(PyObject *obj)
    {
        py_object s(throw_if_not(PyObject_Str(obj)));
        write(s);
    }
    void show(_RuntimeValue *v)
    {
        switch (v->type_) {
        case Extern:
        case ExternAge:
            return write_str(v->cb_arg2);
        case Arg: {
            py_object str(throw_if_not(PyUnicode_FromFormat("arg(%S)", v->cb_arg2)));
            return write(str);
        }
        case Const: {
            py_object obj(rtval_cache(v).to_py());
            return write_str(obj);
        }
        case Add:
            return show_binary(v, " + "_py, v->type_);
        case Sub:
            return show_binary(v, " - "_py, v->type_);
        case Mul:
            return show_binary(v, " * "_py, v->type_);
        case Div:
            return show_binary(v, " / "_py, v->type_);
        case CmpLT:
            return show_binary(v, " < "_py, v->type_);
        case CmpGT:
            return show_binary(v, " > "_py, v->type_);
        case CmpLE:
            return show_binary(v, " <= "_py, v->type_);
        case CmpGE:
            return show_binary(v, " >= "_py, v->type_);
        case CmpEQ:
            return show_binary(v, " == "_py, v->type_);
        case CmpNE:
            return show_binary(v, " != "_py, v->type_);
        case And:
            return show_binary(v, " & "_py, v->type_);
        case Or:
            return show_binary(v, " | "_py, v->type_);
        case Xor:
            return show_binary(v, " ^ "_py, v->type_);
        case Mod:
            return show_binary(v, " % "_py, v->type_);
        case Pow:
            return show_binary(v, "**"_py, v->type_);
        case Not:
            return show_call1(v, "inv"_py);
        case Abs:
            return show_call1(v, "abs"_py);
        case Ceil:
            return show_call1(v, "ceil"_py);
        case Exp:
            return show_call1(v, "exp"_py);
        case Expm1:
            return show_call1(v, "expm1"_py);
        case Floor:
            return show_call1(v, "floor"_py);
        case Log:
            return show_call1(v, "log"_py);
        case Log1p:
            return show_call1(v, "log1p"_py);
        case Log2:
            return show_call1(v, "log2"_py);
        case Log10:
            return show_call1(v, "log10"_py);
        case Sqrt:
            return show_call1(v, "sqrt"_py);
        case Asin:
            return show_call1(v, "arcsin"_py);
        case Acos:
            return show_call1(v, "arccos"_py);
        case Atan:
            return show_call1(v, "arctan"_py);
        case Asinh:
            return show_call1(v, "arcsinh"_py);
        case Acosh:
            return show_call1(v, "arccosh"_py);
        case Atanh:
            return show_call1(v, "arctanh"_py);
        case Sin:
            return show_call1(v, "sin"_py);
        case Cos:
            return show_call1(v, "cos"_py);
        case Tan:
            return show_call1(v, "tan"_py);
        case Sinh:
            return show_call1(v, "sinh"_py);
        case Cosh:
            return show_call1(v, "cosh"_py);
        case Tanh:
            return show_call1(v, "tanh"_py);
        case Rint:
            return show_call1(v, "rint"_py);
        case Max:
            return show_call2(v, "max"_py);
        case Min:
            return show_call2(v, "min"_py);
        case Int64:
            return show_call1(v, "int64"_py);
        case Bool:
            return show_call1(v, "bool"_py);
        case Atan2:
            return show_call2(v, "arctan2"_py);
        case Hypot:
            return show_call2(v, "hypot"_py);
        case Select:
            return show_call3(v, "ifelse"_py);
        default:
            return write("Unknown value"_py);
        }
    }
};

static PyObject *rtvalue_str(PyObject *self)
{
    return py_catch_error([&] {
        rtvalue_printer io;
        io.show((_RuntimeValue*)self);
        return io.getvalue();
    });
}

int rtvalue_init(PyObject *self, PyObject *args, PyObject *kwds)
{
    PyErr_Format(PyExc_TypeError, "RuntimeValue cannot be created directly");
    return -1;
}

static inline void update_rtvalue()
{
    auto type = (PyTypeObject*)RTVal_Type;
    static PyMethodDef rtvalue_array_ufunc_method = {
        "__array_ufunc__", (PyCFunction)(void*)rtvalue_array_ufunc, METH_FASTCALL, 0};
    static PyMethodDef rtvalue_eval_method = {
        "eval", (PyCFunction)(void*)rtvalue_eval, METH_FASTCALL, 0};
    static PyMethodDef rtvalue_ceil_method = {
        "__ceil__", (PyCFunction)(void*)rtvalue_ceil, METH_FASTCALL, 0};
    static PyMethodDef rtvalue_floor_method = {
        "__floor__", (PyCFunction)(void*)rtvalue_floor, METH_FASTCALL, 0};
    static PyMethodDef rtvalue_round_method = {
        "__round__", (PyCFunction)(void*)rtvalue_round, METH_FASTCALL, 0};
    pytype_add_method(type, &rtvalue_array_ufunc_method);
    pytype_add_method(type, &rtvalue_eval_method);
    pytype_add_method(type, &rtvalue_ceil_method);
    pytype_add_method(type, &rtvalue_floor_method);
    pytype_add_method(type, &rtvalue_round_method);
    type->tp_init = rtvalue_init;
    type->tp_repr = rtvalue_str;
    type->tp_str = rtvalue_str;
    type->tp_as_number = &rtvalue_as_number;
    type->tp_richcompare = rtvalue_richcmp;
    PyType_Modified(type);
}

static TagVal rtprop_callback_func(auto *self, unsigned age)
{
    py_object v(throw_if_not(PyObject_GetAttr(self->obj, self->fieldname)));
    if (!is_rtval(v))
        return TagVal::from_py(v);
    auto rv = (_RuntimeValue*)v.get();
    if (rv->type_ == ExternAge && rv->cb_arg2 == (PyObject*)self)
        py_throw_format(PyExc_ValueError, "RT property have not been assigned.");
    rt_eval_cache(rv, age);
    return rtval_cache(rv);
}

template<typename composite_rtprop_data>
static inline __attribute__((returns_nonnull)) composite_rtprop_data*
get_composite_rtprop_data(auto prop, PyObject *obj, PyObject *DataType,
                          composite_rtprop_data*)
{
    auto fieldname = prop->fieldname;
    if (fieldname == Py_None)
        py_throw_format(PyExc_ValueError, "Cannot determine runtime property name");
    if (py_object val(PyObject_GetAttr(obj, fieldname)); !val) {
        PyErr_Clear();
    }
    else if (Py_TYPE(val.get()) == (PyTypeObject*)DataType) {
        return (composite_rtprop_data*)val.release();
    }
    py_object o(pytype_genericalloc(DataType));
    auto data = (composite_rtprop_data*)o.get();
    data->ovr = py_immref(Py_None);
    data->cache = py_immref(Py_None);
    throw_if(PyObject_SetAttr(obj, fieldname, o.get()));
    return (composite_rtprop_data*)o.release();
}

static inline __attribute__((returns_nonnull)) PyObject*
apply_composite_ovr(PyObject *val, PyObject *ovr);

static inline bool
apply_dict_ovr(PyObject *dict, PyObject *k, PyObject *v)
{
    auto field = PyDict_GetItemWithError(dict, k);
    if (field) {
        py_object newfield(apply_composite_ovr(field, v));
        throw_if(PyDict_SetItem(dict, k, newfield));
        return true;
    }
    throw_if(PyErr_Occurred());
    return false;
}

static inline __attribute__((returns_nonnull)) PyObject*
apply_composite_ovr(PyObject *val, PyObject *ovr)
{
    if (!PyDict_Check(ovr))
        return py_newref(ovr);
    if (!PyDict_Size(ovr))
        return py_newref(val);
    if (PyDict_Check(val)) {
        py_object newval(throw_if_not(PyDict_Copy(val)));
        for (auto [k, v]: pydict_iter(ovr)) {
            if (apply_dict_ovr(newval, k, v))
                continue;
            // for scangroup support since only string key is supported
            if (py_object ik(PyNumber_Long(k)); !ik) {
                PyErr_Clear();
            }
            else if (apply_dict_ovr(newval, ik, v)) {
                continue;
            }
            throw_if(PyDict_SetItem(newval, k, v));
        }
        return newval.release();
    }
    if (PyList_Check(val)) {
        py_object newval(throw_if_not(PySequence_List(val)));
        for (auto [k, v]: pydict_iter(ovr)) {
            // for scangroup support since only string key is supported
            py_object ik(throw_if_not(PyNumber_Long(k)));
            auto idx = PyLong_AsLong(ik.get());
            if (idx < 0) {
                throw_if(PyErr_Occurred());
                py_throw_format(PyExc_IndexError, "list index out of range");
            }
            if (idx >= PyList_GET_SIZE(newval.get()))
                py_throw_format(PyExc_IndexError, "list index out of range");
            auto olditem = PyList_GET_ITEM(newval.get(), idx);
            PyList_SET_ITEM(newval.get(), idx, apply_composite_ovr(olditem, v));
            Py_DECREF(olditem);
        }
        return newval.release();
    }
    py_throw_format(PyExc_TypeError, "Unknown value type '%S'", Py_TYPE(val));
}

static inline bool _object_compiled(PyObject *obj)
{
    py_object field(PyObject_GetAttr(obj, "_bb_rt_values"_py));
    if (!field)
        PyErr_Clear();
    return field.get() == Py_None;
}

template<typename composite_rtprop_data>
static inline __attribute__((returns_nonnull)) PyObject*
composite_rtprop_get_res(auto self, PyObject *obj, PyObject *DataType,
                         composite_rtprop_data*)
{
    auto data = get_composite_rtprop_data<composite_rtprop_data>(self, obj,
                                                                 DataType, nullptr);
    py_object py_data((PyObject*)data);
    if (!data->filled || (!data->compiled && _object_compiled(obj))) {
        py_object res(throw_if_not(PyObject_Vectorcall(self->cb, &obj, 1, nullptr)));
        Py_DECREF(data->cache);
        data->cache = res.release();
        data->filled = true;
        data->compiled = _object_compiled(obj);
    }
    if (data->ovr == Py_None)
        return py_newref(data->cache);
    return apply_composite_ovr(data->cache, data->ovr);
}

}
