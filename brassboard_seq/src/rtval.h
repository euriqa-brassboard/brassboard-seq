//

#ifndef BRASSBOARD_SEQ_SRC_RTVAL_H
#define BRASSBOARD_SEQ_SRC_RTVAL_H

#include "Python.h"

namespace brassboard_seq::rtval {

enum ValueType {
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
    // Erf = 40,
    // Erfc = 41,
    // Gamma = 42,
    // Lgamma = 43,
    Rint = 44,
    Max = 45,
    Min = 46,
    Mod = 47,
    // Interp = 48,
    Select = 49,
    // Identity = 50,
    Int64 = 51,
    Bool = 52,
};

static inline ValueType pycmp2valcmp(int op)
{
    assert(op == Py_LT || op == Py_GT || op == Py_LE || op == Py_GE ||
           op == Py_EQ || op == Py_NE);
    switch (op) {
    default:
    case Py_LT: return CmpLT;
    case Py_GT: return CmpGT;
    case Py_LE: return CmpLE;
    case Py_GE: return CmpGE;
    case Py_NE: return CmpNE;
    case Py_EQ: return CmpEQ;
    }
}

}

#endif
