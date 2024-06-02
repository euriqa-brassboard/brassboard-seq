# cython: language_level=3

cdef np # hide import
import numpy as np

cimport cython
from cpython cimport PyErr_Format, PyFloat_AS_DOUBLE

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
