# cython: language_level=3

cdef enum ValueType:
    ExternAge = -2,
    Const = -1,
    Extern = 0,

    Add = 1,
    Sub = 2,
    Mul = 3,
    Div = 4,
    CmpLT = 5,
    CmpGT = 6,
    CmpLE = 7,
    CmpGE = 8,
    CmpNE = 9,
    CmpEQ = 10,
    And = 11,
    Or = 12,
    Xor = 13,
    Not = 14,
    Abs = 15,
    Ceil = 16,
    Exp = 17,
    Expm1 = 18,
    Floor = 19,
    Log = 20,
    Log1p = 21,
    Log2 = 22,
    Log10 = 23,
    Pow = 24,
    Sqrt = 25,
    Asin = 26,
    Acos = 27,
    Atan = 28,
    Atan2 = 29,
    Asinh = 30,
    Acosh = 31,
    Atanh = 32,
    Sin = 33,
    Cos = 34,
    Tan = 35,
    Sinh = 36,
    Cosh = 37,
    Tanh = 38,
    Hypot = 39,
    # Erf = 40,
    # Erfc = 41,
    # Gamma = 42,
    # Lgamma = 43,
    Rint = 44,
    Max = 45,
    Min = 46,
    Mod = 47,
    # Interp = 48,
    Select = 49,
    # Identity = 50,
    Int64 = 51,
    Bool = 52,

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

cdef inline RuntimeValue new_expr3(ValueType type_, RuntimeValue arg0,
                                   RuntimeValue arg1, RuntimeValue arg2):
    self = _new_rtval(type_)
    self.arg0 = arg0
    self.arg1 = arg1
    self.cb_arg2 = arg2
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

cpdef inline bint same_value(v1, v2) noexcept:
    if is_rtval(v1):
        return v1 is v2
    if is_rtval(v2):
        return False
    try:
        return v1 == v2
    except:
        return False

cpdef inline get_value(v, unsigned age):
    if is_rtval(v):
        return rt_eval(<RuntimeValue>v, age)
    return v

cdef class ExternCallback:
    pass
