# cython: language_level=3

cdef io, np # hide import
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
    show(io, write, <RuntimeValue>v.cb_arg2)
    write(')')

cdef void show(io, write, RuntimeValue v):
    cdef ValueType type_ = v.type_
    if type_ == ValueType.Extern:
        write(f'extern({v.cb_arg2})')
    elif type_ == ValueType.ExternAge:
        write(f'extern_age({v.cb_arg2})')
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

cdef np_add = np.add
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

cdef inline object _rt_eval(RuntimeValue self, unsigned age):
    cdef ValueType type_ = self.type_

    if type_ == ValueType.Extern:
        return self.cb_arg2()
    if type_ == ValueType.ExternAge:
        return self.cb_arg2(age)

    arg0 = rt_eval(self.arg0, age)
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
        return rt_eval(self.arg1 if arg0 else <RuntimeValue>self.cb_arg2, age)

    arg1 = rt_eval(self.arg1, age)
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

cdef object rt_eval(RuntimeValue self, unsigned age):
    if self.type_ == ValueType.Const or self.age == age:
        return self.cache
    res = _rt_eval(self, age)
    self.age = age
    self.cache = res
    return res

@cython.final
cdef class RuntimeValue:
    def __init__(self):
        # All instances should be constructed within cython code via
        # `RuntimeValue.__new__` or its wrapper.
        PyErr_Format(TypeError, "RuntimeValue cannot be created directly")

    def eval(self, unsigned age):
        return rt_eval(self, age)

    def __str__(self):
        io = StringIO()
        show(io, io.write, self)
        return io.getvalue()

    def __repr__(self):
        io = StringIO()
        show(io, io.write, self)
        return io.getvalue()
