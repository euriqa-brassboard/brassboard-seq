# cython: language_level=3

import io
cdef StringIO = io.StringIO
import numpy as np

cimport cython
from cpython cimport PyErr_Format, PyFloat_AS_DOUBLE

cdef int operator_precedence(ValueType type_) noexcept:
    if type_ == ValueType.Add or type_ == ValueType.Sub:
        return 3
    if type_ == ValueType.Mul or type_ == ValueType.Div:
        return 2
    if (type_ == ValueType.CmpLT or type_ == ValueType.CmpGT or
        type_ == ValueType.CmpLE or type_ == ValueType.CmpGE):
        return 4
    if type_ == ValueType.CmpNE or type_ == ValueType.CmpEQ:
        return 5
    if type_ == ValueType.And:
        return 6
    if type_ == ValueType.Or:
        return 8
    if type_ == ValueType.Pow:
        return 7
    # if type_ == ValueType.Not: # Not printed as operator anymore
    #     return 1
    return 0

cdef bint needs_parenthesis(RuntimeValue v, ValueType parent_type) noexcept:
    cdef int op_self = operator_precedence(v.type_)
    cdef int op_parent = operator_precedence(parent_type)
    if op_self == 0 or op_parent == 0 or op_self < op_parent:
        return False
    if op_self > op_parent:
        return True
    if parent_type == ValueType.Add or parent_type == ValueType.Mul:
        return False
    return True

cdef void show_arg(io, write, RuntimeValue v, ValueType parent_type):
    cdef bint p = needs_parenthesis(v, parent_type)
    if p:
        write('(')
    show(io, write, v)
    if p:
        write(')')

cdef void show_binary(io, write, RuntimeValue v, str op, ValueType type_):
    show_arg(io, write, v.arg0, type_)
    write(op)
    show_arg(io, write, v.arg1, type_)

cdef void show_call1(io, write, RuntimeValue v, str f):
    write(f)
    write('(')
    show(io, write, v.arg0)
    write(')')

cdef void show_call2(io, write, RuntimeValue v, str f):
    write(f)
    write('(')
    show(io, write, v.arg0)
    write(', ')
    show(io, write, v.arg1)
    write(')')

cdef void show_call3(io, write, RuntimeValue v, str f):
    write(f)
    write('(')
    show(io, write, v.arg0)
    write(', ')
    show(io, write, v.arg1)
    write(', ')
    show(io, write, v.arg2)
    write(')')

cdef void show(io, write, RuntimeValue v):
    cdef ValueType type_ = v.type_
    if type_ == ValueType.Extern:
        write(f'extern({v.cb})')
    elif type_ == ValueType.ExternAge:
        write(f'extern_age({v.cb})')
    elif type_ == ValueType.Const:
        print(v.cache, end='', file=io)
    elif type_ == ValueType.Add:
        show_binary(io, write, v, ' + ', type_)
    elif type_ == ValueType.Sub:
        show_binary(io, write, v, ' - ', type_)
    elif type_ == ValueType.Mul:
        show_binary(io, write, v, ' * ', type_)
    elif type_ == ValueType.Div:
        show_binary(io, write, v, ' / ', type_)
    elif type_ == ValueType.CmpLT:
        show_binary(io, write, v, ' < ', type_)
    elif type_ == ValueType.CmpGT:
        show_binary(io, write, v, ' > ', type_)
    elif type_ == ValueType.CmpLE:
        show_binary(io, write, v, ' <= ', type_)
    elif type_ == ValueType.CmpGE:
        show_binary(io, write, v, ' >= ', type_)
    elif type_ == ValueType.CmpEQ:
        show_binary(io, write, v, ' == ', type_)
    elif type_ == ValueType.CmpNE:
        show_binary(io, write, v, ' != ', type_)
    elif type_ == ValueType.And:
        show_binary(io, write, v, ' & ', type_)
    elif type_ == ValueType.Or:
        show_binary(io, write, v, ' | ', type_)
    elif type_ == ValueType.Xor:
        show_binary(io, write, v, ' ^ ', type_)
    elif type_ == ValueType.Mod:
        show_binary(io, write, v, ' % ', type_)
    elif type_ == ValueType.Pow:
        show_binary(io, write, v, '**', type_)
    elif type_ == ValueType.Not:
        show_call1(io, write, v, 'inv')
    elif type_ == ValueType.Abs:
        show_call1(io, write, v, 'abs')
    elif type_ == ValueType.Ceil:
        show_call1(io, write, v, 'ceil')
    elif type_ == ValueType.Exp:
        show_call1(io, write, v, 'exp')
    elif type_ == ValueType.Expm1:
        show_call1(io, write, v, 'expm1')
    elif type_ == ValueType.Floor:
        show_call1(io, write, v, 'floor')
    elif type_ == ValueType.Log:
        show_call1(io, write, v, 'log')
    elif type_ == ValueType.Log1p:
        show_call1(io, write, v, 'log1p')
    elif type_ == ValueType.Log2:
        show_call1(io, write, v, 'log2')
    elif type_ == ValueType.Log10:
        show_call1(io, write, v, 'log10')
    elif type_ == ValueType.Sqrt:
        show_call1(io, write, v, 'sqrt')
    elif type_ == ValueType.Asin:
        show_call1(io, write, v, 'arcsin')
    elif type_ == ValueType.Acos:
        show_call1(io, write, v, 'arccos')
    elif type_ == ValueType.Atan:
        show_call1(io, write, v, 'arctan')
    elif type_ == ValueType.Asinh:
        show_call1(io, write, v, 'arcsinh')
    elif type_ == ValueType.Acosh:
        show_call1(io, write, v, 'arccosh')
    elif type_ == ValueType.Atanh:
        show_call1(io, write, v, 'arctanh')
    elif type_ == ValueType.Sin:
        show_call1(io, write, v, 'sin')
    elif type_ == ValueType.Cos:
        show_call1(io, write, v, 'cos')
    elif type_ == ValueType.Tan:
        show_call1(io, write, v, 'tan')
    elif type_ == ValueType.Sinh:
        show_call1(io, write, v, 'sinh')
    elif type_ == ValueType.Cosh:
        show_call1(io, write, v, 'cosh')
    elif type_ == ValueType.Tanh:
        show_call1(io, write, v, 'tanh')
    elif type_ == ValueType.Rint:
        show_call1(io, write, v, 'rint')
    elif type_ == ValueType.Max:
        show_call2(io, write, v, 'max')
    elif type_ == ValueType.Min:
        show_call2(io, write, v, 'min')
    elif type_ == ValueType.Int64:
        show_call1(io, write, v, 'int64')
    elif type_ == ValueType.Bool:
        show_call1(io, write, v, 'bool')
    elif type_ == ValueType.Atan2:
        show_call2(io, write, v, 'arctan2')
    elif type_ == ValueType.Hypot:
        show_call2(io, write, v, 'hypot')
    elif type_ == ValueType.Select:
        show_call3(io, write, v, 'ifelse')
    else:
        write('Unknown value')

cdef extern from *:
    """
    #include "Python.h"

    static inline PyObject *_add_or_sub(PyObject *a, PyObject *b, bool issub)
    {
        if (issub) {
            return PyNumber_Subtract(a, b);
        }
        else {
            return PyNumber_Add(a, b);
        }
    }
    static inline int pycmp2valcmp(int op)
    {
        switch (op) {
        case Py_LT: return 5;
        case Py_GT: return 6;
        case Py_LE: return 7;
        case Py_GE: return 8;
        case Py_NE: return 9;
        case Py_EQ: return 10;
        default: return 0;
        }
    }
    """
    object _add_or_sub(object a, object b, bint issub)
    int pycmp2valcmp(int op) noexcept

cdef _new_addsub(c, RuntimeValue v, bint s):
    if c == 0 and not s:
        return v
    return new_expr2(ValueType.Sub if s else ValueType.Add, new_const(c), v)

cdef _build_addsub(v0, v1, bint issub):
    cdef bint ns0 = False
    cdef bint ns1 = False
    cdef RuntimeValue nv0
    cdef RuntimeValue nv1
    cdef ValueType type_
    if not is_rtval(v0):
        nc = v0
        nv0 = None
    else:
        nc = False
        nv0 = <RuntimeValue>v0
        type_ = nv0.type_
        if type_ == ValueType.Const:
            nc = nv0.cache
            nv0 = None
        elif type_ == ValueType.Add:
            arg0 = <RuntimeValue>nv0.arg0
            arg1 = <RuntimeValue>nv0.arg1
            if arg0.type_ == ValueType.Const:
                nc = arg0.cache
                nv0 = arg1
            elif arg1.type_ == ValueType.Const:
                nc = arg1.cache
                nv0 = arg0
        elif type_ == ValueType.Sub:
            arg0 = <RuntimeValue>nv0.arg0
            arg1 = <RuntimeValue>nv0.arg1
            if arg0.type_ == ValueType.Const:
                ns0 = True
                nc = arg0.cache
                nv0 = arg1
            elif arg1.type_ == ValueType.Const:
                nc = -arg1.cache
                nv0 = arg0
    if not is_rtval(v1):
        nc = _add_or_sub(nc, v1, issub)
        nv1 = None
    else:
        nv1 = <RuntimeValue>v1
        type_ = nv1.type_
        if type_ == ValueType.Const:
            nc = _add_or_sub(nc, nv1.cache, issub)
            nv1 = None
        elif type_ == ValueType.Add:
            arg0 = <RuntimeValue>nv1.arg0
            arg1 = <RuntimeValue>nv1.arg1
            if arg0.type_ == ValueType.Const:
                nc = _add_or_sub(nc, arg0.cache, issub)
                nv1 = arg1
            elif arg1.type_ == ValueType.Const:
                nc = _add_or_sub(nc, arg1.cache, issub)
                nv1 = arg0
        elif type_ == ValueType.Sub:
            arg0 = <RuntimeValue>nv1.arg0
            arg1 = <RuntimeValue>nv1.arg1
            if arg0.type_ == ValueType.Const:
                ns1 = True
                nc = _add_or_sub(nc, arg0.cache, issub)
                nv1 = arg1
            elif arg1.type_ == ValueType.Const:
                nc = _add_or_sub(nc, arg1.cache, not issub)
                nv1 = arg0
    if nv0 is v0 and v1 is nv1:
        return new_expr2(ValueType.Sub if issub else ValueType.Add, v0, v1)
    if issub:
        ns1 = not ns1
    if nv0 is None:
        if nv1 is None:
            return new_const(nc)
        return _new_addsub(nc, nv1, ns1)
    if nv1 is None:
        return _new_addsub(nc, nv0, ns0)
    cdef bint ns = False
    if ns0:
        if ns1:
            nv = new_expr2(ValueType.Add, nv0, nv1)
            ns = True
        else:
            nv = new_expr2(ValueType.Sub, nv1, nv0)
    elif ns1:
        nv = new_expr2(ValueType.Sub, nv0, nv1)
    else:
        nv = new_expr2(ValueType.Add, nv0, nv1)
    return _new_addsub(nc, nv, ns)

cdef RuntimeValue wrap_value(v):
    if is_rtval(v):
        return <RuntimeValue>v
    return new_const(v)

cdef np_add = np.add
cdef np_subtract = np.subtract
cdef np_multiply = np.multiply
cdef np_divide = np.divide
cdef np_remainder = np.remainder
cdef np_bitwise_and = np.bitwise_and
cdef np_bitwise_or = np.bitwise_or
cdef np_bitwise_xor = np.bitwise_xor
cdef np_power = np.power
cdef np_less = np.less
cdef np_greater = np.greater
cdef np_less_equal = np.less_equal
cdef np_greater_equal = np.greater_equal
cdef np_equal = np.equal
cdef np_not_equal = np.not_equal

cdef np_abs = np.abs
cdef np_ceil = np.ceil
cdef np_exp = np.exp
cdef np_expm1 = np.expm1
cdef np_floor = np.floor
cdef np_log = np.log
cdef np_log1p = np.log1p
cdef np_log2 = np.log2
cdef np_log10 = np.log10
cdef np_sqrt = np.sqrt
cdef np_arcsin = np.arcsin
cdef np_arccos = np.arccos
cdef np_arctan = np.arctan
cdef np_arctan2 = np.arctan2
cdef np_arcsinh = np.arcsinh
cdef np_arccosh = np.arccosh
cdef np_arctanh = np.arctanh
cdef np_sin = np.sin
cdef np_cos = np.cos
cdef np_tan = np.tan
cdef np_sinh = np.sinh
cdef np_cosh = np.cosh
cdef np_tanh = np.tanh
cdef np_hypot = np.hypot
cdef np_rint = np.rint

cdef inline _round_int64(v):
    if type(v) is int:
        return v
    cdef long long vi
    if type(v) is float:
        vf = PyFloat_AS_DOUBLE(v)
        vi = <long long>(vf + 0.5 if vf >= 0 else vf - 0.5)
        return vi
    return int(round(v))

@cython.c_api_binop_methods(True)
@cython.final
cdef class RuntimeValue:
    def __init__(self):
        # All instances should be constructed within cython code via
        # `RuntimeValue.__new__` or its wrapper.
        PyErr_Format(TypeError, "RuntimeValue cannot be created directly")

    cpdef object eval(self, long long age):
        if self.type_ == ValueType.Const or self.age == age:
            return self.cache
        res = self._eval(age)
        self.age = age
        self.cache = res
        return res

    cdef object _eval(self, long long age):
        cdef ValueType type_ = self.type_

        if type_ == ValueType.Extern:
            return self.cb()
        if type_ == ValueType.ExternAge:
            return self.cb(age)

        arg0 = (<RuntimeValue>self.arg0).eval(age)
        if type_ == ValueType.Not:
            if type(arg0) is bool:
                return arg0 is False
            return ~arg0
        if type_ == ValueType.Abs:
            return abs(arg0)
        if type_ == ValueType.Ceil:
            return np_ceil(arg0)
        if type_ == ValueType.Exp:
            return np_exp(arg0)
        if type_ == ValueType.Expm1:
            return np_expm1(arg0)
        if type_ == ValueType.Floor:
            return np_floor(arg0)
        if type_ == ValueType.Log:
            return np_log(arg0)
        if type_ == ValueType.Log1p:
            return np_log1p(arg0)
        if type_ == ValueType.Log2:
            return np_log2(arg0)
        if type_ == ValueType.Log10:
            return np_log10(arg0)
        if type_ == ValueType.Sqrt:
            return np_sqrt(arg0)
        if type_ == ValueType.Asin:
            return np_arcsin(arg0)
        if type_ == ValueType.Acos:
            return np_arccos(arg0)
        if type_ == ValueType.Atan:
            return np_arctan(arg0)
        if type_ == ValueType.Asinh:
            return np_arcsinh(arg0)
        if type_ == ValueType.Acosh:
            return np_arccosh(arg0)
        if type_ == ValueType.Atanh:
            return np_arctanh(arg0)
        if type_ == ValueType.Sin:
            return np_sin(arg0)
        if type_ == ValueType.Cos:
            return np_cos(arg0)
        if type_ == ValueType.Tan:
            return np_tan(arg0)
        if type_ == ValueType.Sinh:
            return np_sinh(arg0)
        if type_ == ValueType.Cosh:
            return np_cosh(arg0)
        if type_ == ValueType.Tanh:
            return np_tanh(arg0)
        if type_ == ValueType.Rint:
            return np_rint(arg0)
        if type_ == ValueType.Int64:
            return _round_int64(arg0)
        if type_ == ValueType.Bool:
            return bool(arg0)

        if type_ == ValueType.Select:
            return (<RuntimeValue>(self.arg1 if arg0 else self.arg2)).eval(age)

        arg1 = (<RuntimeValue>self.arg1).eval(age)
        if type_ == ValueType.Add:
            return arg0 + arg1
        if type_ == ValueType.Sub:
            return arg0 - arg1
        if type_ == ValueType.Mul:
            return arg0 * arg1
        if type_ == ValueType.Div:
            return arg0 / arg1
        if type_ == ValueType.CmpLT:
            return arg0 < arg1
        if type_ == ValueType.CmpGT:
            return arg0 > arg1
        if type_ == ValueType.CmpLE:
            return arg0 <= arg1
        if type_ == ValueType.CmpGE:
            return arg0 >= arg1
        if type_ == ValueType.CmpNE:
            return arg0 != arg1
        if type_ == ValueType.CmpEQ:
            return arg0 == arg1
        if type_ == ValueType.And:
            return arg0 & arg1
        if type_ == ValueType.Or:
            return arg0 | arg1
        if type_ == ValueType.Xor:
            return arg0 ^ arg1
        if type_ == ValueType.Pow:
            return arg0**arg1
        if type_ == ValueType.Hypot:
            return np_hypot(arg0, arg1)
        if type_ == ValueType.Atan2:
            return np_arctan2(arg0, arg1)
        if type_ == ValueType.Mod:
            return arg0 % arg1
        if type_ == ValueType.Max:
            return max(arg0, arg1)
        if type_ == ValueType.Min:
            return min(arg0, arg1)
        PyErr_Format(ValueError, 'Unknown value type')

    def __str__(self):
        io = StringIO()
        show(io, io.write, self)
        return io.getvalue()

    def __repr__(self):
        io = StringIO()
        show(io, io.write, self)
        return io.getvalue()

    def __add__(self, other):
        return _build_addsub(self, other, False)
    def __sub__(self, other):
        return _build_addsub(self, other, True)

    def __mul__(self, other):
        return new_expr2(ValueType.Mul, wrap_value(self), wrap_value(other))

    def __truediv__(self, other):
        return new_expr2(ValueType.Div, wrap_value(self), wrap_value(other))

    def __and__(self, other):
        return new_expr2(ValueType.And, wrap_value(self), wrap_value(other))

    def __or__(self, other):
        return new_expr2(ValueType.Or, wrap_value(self), wrap_value(other))

    def __xor__(self, other):
        return new_expr2(ValueType.Xor, wrap_value(self), wrap_value(other))

    def __pow__(self, other):
        return new_expr2(ValueType.Pow, wrap_value(self), wrap_value(other))

    def __mod__(self, other):
        return new_expr2(ValueType.Mod, wrap_value(self), wrap_value(other))

    def __pos__(self):
        return self
    def __neg__(self):
        return _build_addsub(0, self, True)

    def __richcmp__(self, other, int op):
        cdef int typ = pycmp2valcmp(op)
        if typ == 0:
            return NotImplemented
        cdef RuntimeValue v2
        if is_rtval(other):
            if self is other:
                return (typ == ValueType.CmpLE or typ == ValueType.CmpGE or
                        typ == ValueType.CmpEQ)
            v2 = <RuntimeValue>other
        else:
            v2 = new_const(other)
        return new_expr2(<ValueType>typ, self, v2)

    def __abs__(self):
        if self.type_ == ValueType.Abs:
            return self
        return new_expr1(ValueType.Abs, self)

    def __ceil__(self):
        if self.type_ == ValueType.Ceil:
            return self
        return new_expr1(ValueType.Ceil, self)

    def __floor__(self):
        if self.type_ == ValueType.Floor:
            return self
        return new_expr1(ValueType.Floor, self)

    # Artifically limit the supported ufunc
    # in case we need to do any processing later
    # (e.g. compiling/sending it to kernel etc).
    def __array_ufunc__(self, ufunc, methods, *inputs):
        if methods != '__call__':
            return NotImplemented
        # Needed for numpy type support
        if ufunc is np_add:
            return _build_addsub(inputs[0], inputs[1], False)
        if ufunc is np_subtract:
            return _build_addsub(inputs[0], inputs[1], True)
        if ufunc is np_multiply:
            return new_expr2(ValueType.Mul,
                             wrap_value(inputs[0]), wrap_value(inputs[1]))
        if ufunc is np_divide:
            return new_expr2(ValueType.Div,
                             wrap_value(inputs[0]), wrap_value(inputs[1]))
        if ufunc is np_remainder:
            return new_expr2(ValueType.Mod,
                             wrap_value(inputs[0]), wrap_value(inputs[1]))
        if ufunc is np_bitwise_and:
            return new_expr2(ValueType.And,
                             wrap_value(inputs[0]), wrap_value(inputs[1]))
        if ufunc is np_bitwise_or:
            return new_expr2(ValueType.Or,
                             wrap_value(inputs[0]), wrap_value(inputs[1]))
        if ufunc is np_bitwise_xor:
            return new_expr2(ValueType.Xor,
                             wrap_value(inputs[0]), wrap_value(inputs[1]))
        if ufunc is np_power:
            return new_expr2(ValueType.Pow,
                             wrap_value(inputs[0]), wrap_value(inputs[1]))
        if ufunc is np_less:
            return new_expr2(ValueType.CmpLT,
                             wrap_value(inputs[0]), wrap_value(inputs[1]))
        if ufunc is np_greater:
            return new_expr2(ValueType.CmpGT,
                             wrap_value(inputs[0]), wrap_value(inputs[1]))
        if ufunc is np_less_equal:
            return new_expr2(ValueType.CmpLE,
                             wrap_value(inputs[0]), wrap_value(inputs[1]))
        if ufunc is np_greater_equal:
            return new_expr2(ValueType.CmpGE,
                             wrap_value(inputs[0]), wrap_value(inputs[1]))
        if ufunc is np_equal:
            return new_expr2(ValueType.CmpEQ,
                             wrap_value(inputs[0]), wrap_value(inputs[1]))
        if ufunc is np_not_equal:
            return new_expr2(ValueType.CmpNE,
                             wrap_value(inputs[0]), wrap_value(inputs[1]))
        if ufunc is np_abs:
            arg0 = <RuntimeValue?>inputs[0]
            if arg0.type_ == ValueType.Abs:
                return arg0
            return new_expr1(ValueType.Abs, arg0)
        if ufunc is np_ceil:
            arg0 = <RuntimeValue?>inputs[0]
            if arg0.type_ == ValueType.Ceil:
                return arg0
            return new_expr1(ValueType.Ceil, arg0)
        if ufunc is np_exp:
            return new_expr1(ValueType.Exp, <RuntimeValue?>inputs[0])
        if ufunc is np_expm1:
            return new_expr1(ValueType.Expm1, <RuntimeValue?>inputs[0])
        if ufunc is np_floor:
            arg0 = <RuntimeValue?>inputs[0]
            if arg0.type_ == ValueType.Floor:
                return arg0
            return new_expr1(ValueType.Floor, arg0)
        if ufunc is np_log:
            return new_expr1(ValueType.Log, <RuntimeValue?>inputs[0])
        if ufunc is np_log1p:
            return new_expr1(ValueType.Log1p, <RuntimeValue?>inputs[0])
        if ufunc is np_log2:
            return new_expr1(ValueType.Log2, <RuntimeValue?>inputs[0])
        if ufunc is np_log10:
            return new_expr1(ValueType.Log10, <RuntimeValue?>inputs[0])
        if ufunc is np_sqrt:
            return new_expr1(ValueType.Sqrt, <RuntimeValue?>inputs[0])
        if ufunc is np_arcsin:
            return new_expr1(ValueType.Asin, <RuntimeValue?>inputs[0])
        if ufunc is np_arccos:
            return new_expr1(ValueType.Acos, <RuntimeValue?>inputs[0])
        if ufunc is np_arctan:
            return new_expr1(ValueType.Atan, <RuntimeValue?>inputs[0])
        if ufunc is np_arctan2:
            return new_expr2(ValueType.Atan2,
                             wrap_value(inputs[0]), wrap_value(inputs[1]))
        if ufunc is np_arcsinh:
            return new_expr1(ValueType.Asinh, <RuntimeValue?>inputs[0])
        if ufunc is np_arccosh:
            return new_expr1(ValueType.Acosh, <RuntimeValue?>inputs[0])
        if ufunc is np_arctanh:
            return new_expr1(ValueType.Atanh, <RuntimeValue?>inputs[0])
        if ufunc is np_sin:
            return new_expr1(ValueType.Sin, <RuntimeValue?>inputs[0])
        if ufunc is np_cos:
            return new_expr1(ValueType.Cos, <RuntimeValue?>inputs[0])
        if ufunc is np_tan:
            return new_expr1(ValueType.Tan, <RuntimeValue?>inputs[0])
        if ufunc is np_sinh:
            return new_expr1(ValueType.Sinh, <RuntimeValue?>inputs[0])
        if ufunc is np_cosh:
            return new_expr1(ValueType.Cosh, <RuntimeValue?>inputs[0])
        if ufunc is np_tanh:
            return new_expr1(ValueType.Tanh, <RuntimeValue?>inputs[0])
        if ufunc is np_hypot:
            return new_expr2(ValueType.Hypot,
                             wrap_value(inputs[0]), wrap_value(inputs[1]))
        if ufunc is np_rint:
            arg0 = <RuntimeValue?>inputs[0]
            if arg0.type_ == ValueType.Rint:
                return arg0
            return new_expr1(ValueType.Rint, arg0)
        return NotImplemented

cpdef inv(v):
    if type(v) is bool:
        return v is False
    cdef RuntimeValue _v
    if is_rtval(v):
        _v = <RuntimeValue>v
        if _v.type_ == ValueType.Not:
            return _v.arg0
        return new_expr1(ValueType.Not, _v)
    return ~v

cpdef convert_bool(_v):
    cdef RuntimeValue v
    if is_rtval(_v):
        v = <RuntimeValue>(_v)
        if v.type_ == ValueType.Int64:
            v = v.arg0
        if v.type_ == ValueType.Bool:
            return v
        return new_expr1(ValueType.Bool, v)
    return bool(_v)

cpdef round_int64(_v):
    cdef RuntimeValue v
    if is_rtval(_v):
        return round_int64_rt(<RuntimeValue>_v)
    return _round_int64(_v)

cpdef ifelse(b, v1, v2):
    if is_rtval(b):
        return new_expr3(ValueType.Select, b, wrap_value(v1), wrap_value(v2))
    return v1 if b else v2

cpdef min_val(v1, v2):
    if not is_rtval(v1) and not is_rtval(v2):
        return min(v1, v2)
    if v1 is v2:
        return v1
    return new_expr2(ValueType.Min, wrap_value(v1), wrap_value(v2))

cpdef max_val(v1, v2):
    if not is_rtval(v1) and not is_rtval(v2):
        return max(v1, v2)
    if v1 is v2:
        return v1
    return new_expr2(ValueType.Max, wrap_value(v1), wrap_value(v2))

@cython.final
cdef class rtprop_callback:
    cdef obj
    cdef str fieldname

    def __init__(self):
        PyErr_Format(TypeError, "rtprop_callback cannot be created directly")

    def __call__(self, long long age):
        _v = getattr(self.obj, self.fieldname)
        if not is_rtval(_v):
            return _v
        cdef RuntimeValue v = <RuntimeValue>_v
        if (v.type_ == ValueType.ExternAge and v.cb is self):
            PyErr_Format(ValueError, 'RT property have not been assigned.')
        return v.eval(age)

cdef rtprop_callback new_rtprop_callback(obj, str fieldname):
    self = <rtprop_callback>rtprop_callback.__new__(rtprop_callback)
    self.obj = obj
    self.fieldname = fieldname
    return self

@cython.final
cdef class RTProp:
    cdef str fieldname

    cdef __init_class(self, cls):
        if self.fieldname is not None:
            return
        for _name in dir(cls):
            name = <str>_name
            field = getattr(cls, name)
            if isinstance(field, RTProp):
                (<RTProp>field).fieldname = '_RTProp_value_' + name
        if self.fieldname is None:
            PyErr_Format(ValueError, 'Cannot determine runtime property name')

    def __set__(self, obj, value):
        self.__init_class(type(obj))
        setattr(obj, self.fieldname, value)

    def __get__(self, obj, objtype):
        if obj is None:
            return self
        self.__init_class(type(obj))
        fieldname = self.fieldname
        try:
            return getattr(obj, fieldname)
        except AttributeError:
            pass
        value = new_extern_age(new_rtprop_callback(obj, fieldname))
        setattr(obj, fieldname, value)
        return value
