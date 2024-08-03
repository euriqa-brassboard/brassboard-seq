# cython: language_level=3

cdef extern from "src/rtval.h" namespace "brassboard_seq::rtval":
    cdef enum ValueType:
        ExternAge
        Const
        Extern

        Add
        Sub
        Mul
        Div
        CmpLT
        CmpGT
        CmpLE
        CmpGE
        CmpNE
        CmpEQ
        And
        Or
        Xor
        Not
        Abs
        Ceil
        Exp
        Expm1
        Floor
        Log
        Log1p
        Log2
        Log10
        Pow
        Sqrt
        Asin
        Acos
        Atan
        Atan2
        Asinh
        Acosh
        Atanh
        Sin
        Cos
        Tan
        Sinh
        Cosh
        Tanh
        Hypot
        # Erf
        # Erfc
        # Gamma
        # Lgamma
        Rint
        Max
        Min
        Mod
        # Interp
        Select
        # Identity
        Int64
        Bool

    ValueType pycmp2valcmp(int op) noexcept

cdef class RuntimeValue:
    cdef ValueType type_
    cdef unsigned age
    cdef RuntimeValue arg0
    cdef RuntimeValue arg1
    cdef object cb_arg2
    cdef object cache

cdef rt_eval(RuntimeValue self, unsigned age)

cdef inline RuntimeValue _new_rtval(ValueType type_):
    # Avoid passing arguments to the constructor which requires the arguments
    # to be boxed and put in a tuple. This improves the performance by about 20%.
    self = <RuntimeValue>RuntimeValue.__new__(RuntimeValue)
    self.type_ = type_
    self.age = -1
    return self

cpdef inline RuntimeValue new_const(v):
    self = _new_rtval(ValueType.Const)
    self.cache = v
    return self

cpdef inline RuntimeValue new_extern(cb):
    self = _new_rtval(ValueType.Extern)
    self.cb_arg2 = cb
    return self

cpdef inline RuntimeValue new_extern_age(cb):
    self = _new_rtval(ValueType.ExternAge)
    self.cb_arg2 = cb
    return self

cdef inline RuntimeValue new_expr1(ValueType type_, RuntimeValue arg0):
    self = _new_rtval(type_)
    self.arg0 = arg0
    return self

cdef inline RuntimeValue new_expr2(ValueType type_, RuntimeValue arg0,
                                   RuntimeValue arg1):
    self = _new_rtval(type_)
    self.arg0 = arg0
    self.arg1 = arg1
    return self

cdef inline RuntimeValue round_int64_rt(RuntimeValue v):
    if v.type_ == ValueType.Int64:
        return v
    return new_expr1(ValueType.Int64, v)

cpdef inv(v)
cpdef convert_bool(v)
cpdef round_int64(v)
cpdef ifelse(b, v1, v2)

cdef inline bint is_rtval(v) noexcept:
    return type(v) is RuntimeValue

cpdef inline get_value(v, unsigned age):
    if is_rtval(v):
        return rt_eval(<RuntimeValue>v, age)
    return v

cdef class ExternCallback:
    pass
