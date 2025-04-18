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

GEN_BIN_OP(Add, Bool, Bool)
GEN_BIN_OP(Add, Bool, Int64)
GEN_BIN_OP(Add, Bool, Float64)
GEN_BIN_OP(Add, Int64, Bool)
GEN_BIN_OP(Add, Int64, Int64)
GEN_BIN_OP(Add, Int64, Float64)
GEN_BIN_OP(Add, Float64, Bool)
GEN_BIN_OP(Add, Float64, Int64)
GEN_BIN_OP(Add, Float64, Float64)
GEN_BIN_OP(Sub, Bool, Bool)
GEN_BIN_OP(Sub, Bool, Int64)
GEN_BIN_OP(Sub, Bool, Float64)
GEN_BIN_OP(Sub, Int64, Bool)
GEN_BIN_OP(Sub, Int64, Int64)
GEN_BIN_OP(Sub, Int64, Float64)
GEN_BIN_OP(Sub, Float64, Bool)
GEN_BIN_OP(Sub, Float64, Int64)
GEN_BIN_OP(Sub, Float64, Float64)
GEN_BIN_OP(Mul, Bool, Bool)
GEN_BIN_OP(Mul, Bool, Int64)
GEN_BIN_OP(Mul, Bool, Float64)
GEN_BIN_OP(Mul, Int64, Bool)
GEN_BIN_OP(Mul, Int64, Int64)
GEN_BIN_OP(Mul, Int64, Float64)
GEN_BIN_OP(Mul, Float64, Bool)
GEN_BIN_OP(Mul, Float64, Int64)
GEN_BIN_OP(Mul, Float64, Float64)
GEN_BIN_OP(Div, Bool, Bool)
GEN_BIN_OP(Div, Bool, Int64)
GEN_BIN_OP(Div, Bool, Float64)
GEN_BIN_OP(Div, Int64, Bool)
GEN_BIN_OP(Div, Int64, Int64)
GEN_BIN_OP(Div, Int64, Float64)
GEN_BIN_OP(Div, Float64, Bool)
GEN_BIN_OP(Div, Float64, Int64)
GEN_BIN_OP(Div, Float64, Float64)
GEN_BIN_OP(Mod, Bool, Bool)
GEN_BIN_OP(Mod, Bool, Int64)
GEN_BIN_OP(Mod, Bool, Float64)
GEN_BIN_OP(Mod, Int64, Bool)
GEN_BIN_OP(Mod, Int64, Int64)
GEN_BIN_OP(Mod, Int64, Float64)
GEN_BIN_OP(Mod, Float64, Bool)
GEN_BIN_OP(Mod, Float64, Int64)
GEN_BIN_OP(Mod, Float64, Float64)
GEN_BIN_OP(Min, Bool, Bool)
GEN_BIN_OP(Min, Bool, Int64)
GEN_BIN_OP(Min, Bool, Float64)
GEN_BIN_OP(Min, Int64, Bool)
GEN_BIN_OP(Min, Int64, Int64)
GEN_BIN_OP(Min, Int64, Float64)
GEN_BIN_OP(Min, Float64, Bool)
GEN_BIN_OP(Min, Float64, Int64)
GEN_BIN_OP(Min, Float64, Float64)
GEN_BIN_OP(Max, Bool, Bool)
GEN_BIN_OP(Max, Bool, Int64)
GEN_BIN_OP(Max, Bool, Float64)
GEN_BIN_OP(Max, Int64, Bool)
GEN_BIN_OP(Max, Int64, Int64)
GEN_BIN_OP(Max, Int64, Float64)
GEN_BIN_OP(Max, Float64, Bool)
GEN_BIN_OP(Max, Float64, Int64)
GEN_BIN_OP(Max, Float64, Float64)
GEN_BIN_OP(CmpLT, Bool, Bool)
GEN_BIN_OP(CmpLT, Bool, Int64)
GEN_BIN_OP(CmpLT, Bool, Float64)
GEN_BIN_OP(CmpLT, Int64, Bool)
GEN_BIN_OP(CmpLT, Int64, Int64)
GEN_BIN_OP(CmpLT, Int64, Float64)
GEN_BIN_OP(CmpLT, Float64, Bool)
GEN_BIN_OP(CmpLT, Float64, Int64)
GEN_BIN_OP(CmpLT, Float64, Float64)
GEN_BIN_OP(CmpGT, Bool, Bool)
GEN_BIN_OP(CmpGT, Bool, Int64)
GEN_BIN_OP(CmpGT, Bool, Float64)
GEN_BIN_OP(CmpGT, Int64, Bool)
GEN_BIN_OP(CmpGT, Int64, Int64)
GEN_BIN_OP(CmpGT, Int64, Float64)
GEN_BIN_OP(CmpGT, Float64, Bool)
GEN_BIN_OP(CmpGT, Float64, Int64)
GEN_BIN_OP(CmpGT, Float64, Float64)
GEN_BIN_OP(CmpLE, Bool, Bool)
GEN_BIN_OP(CmpLE, Bool, Int64)
GEN_BIN_OP(CmpLE, Bool, Float64)
GEN_BIN_OP(CmpLE, Int64, Bool)
GEN_BIN_OP(CmpLE, Int64, Int64)
GEN_BIN_OP(CmpLE, Int64, Float64)
GEN_BIN_OP(CmpLE, Float64, Bool)
GEN_BIN_OP(CmpLE, Float64, Int64)
GEN_BIN_OP(CmpLE, Float64, Float64)
GEN_BIN_OP(CmpGE, Bool, Bool)
GEN_BIN_OP(CmpGE, Bool, Int64)
GEN_BIN_OP(CmpGE, Bool, Float64)
GEN_BIN_OP(CmpGE, Int64, Bool)
GEN_BIN_OP(CmpGE, Int64, Int64)
GEN_BIN_OP(CmpGE, Int64, Float64)
GEN_BIN_OP(CmpGE, Float64, Bool)
GEN_BIN_OP(CmpGE, Float64, Int64)
GEN_BIN_OP(CmpGE, Float64, Float64)
GEN_BIN_OP(CmpEQ, Bool, Bool)
GEN_BIN_OP(CmpEQ, Bool, Int64)
GEN_BIN_OP(CmpEQ, Bool, Float64)
GEN_BIN_OP(CmpEQ, Int64, Bool)
GEN_BIN_OP(CmpEQ, Int64, Int64)
GEN_BIN_OP(CmpEQ, Int64, Float64)
GEN_BIN_OP(CmpEQ, Float64, Bool)
GEN_BIN_OP(CmpEQ, Float64, Int64)
GEN_BIN_OP(CmpEQ, Float64, Float64)
GEN_BIN_OP(CmpNE, Bool, Bool)
GEN_BIN_OP(CmpNE, Bool, Int64)
GEN_BIN_OP(CmpNE, Bool, Float64)
GEN_BIN_OP(CmpNE, Int64, Bool)
GEN_BIN_OP(CmpNE, Int64, Int64)
GEN_BIN_OP(CmpNE, Int64, Float64)
GEN_BIN_OP(CmpNE, Float64, Bool)
GEN_BIN_OP(CmpNE, Float64, Int64)
GEN_BIN_OP(CmpNE, Float64, Float64)
GEN_BIN_OP(Atan2, Bool, Bool)
GEN_BIN_OP(Atan2, Bool, Int64)
GEN_BIN_OP(Atan2, Bool, Float64)
GEN_BIN_OP(Atan2, Int64, Bool)
GEN_BIN_OP(Atan2, Int64, Int64)
GEN_BIN_OP(Atan2, Int64, Float64)
GEN_BIN_OP(Atan2, Float64, Bool)
GEN_BIN_OP(Atan2, Float64, Int64)
GEN_BIN_OP(Atan2, Float64, Float64)
GEN_BIN_OP(Hypot, Bool, Bool)
GEN_BIN_OP(Hypot, Bool, Int64)
GEN_BIN_OP(Hypot, Bool, Float64)
GEN_BIN_OP(Hypot, Int64, Bool)
GEN_BIN_OP(Hypot, Int64, Int64)
GEN_BIN_OP(Hypot, Int64, Float64)
GEN_BIN_OP(Hypot, Float64, Bool)
GEN_BIN_OP(Hypot, Float64, Int64)
GEN_BIN_OP(Hypot, Float64, Float64)
GEN_BIN_OP(Pow, Bool, Bool)
GEN_BIN_OP(Pow, Bool, Int64)
GEN_BIN_OP(Pow, Bool, Float64)
GEN_BIN_OP(Pow, Int64, Bool)
GEN_BIN_OP(Pow, Int64, Int64)
GEN_BIN_OP(Pow, Int64, Float64)
GEN_BIN_OP(Pow, Float64, Bool)
GEN_BIN_OP(Pow, Float64, Int64)
GEN_BIN_OP(Pow, Float64, Float64)
GEN_BIN_OP(And, Bool, Bool)
GEN_BIN_OP(And, Bool, Int64)
GEN_BIN_OP(And, Bool, Float64)
GEN_BIN_OP(And, Int64, Bool)
GEN_BIN_OP(And, Int64, Int64)
GEN_BIN_OP(And, Int64, Float64)
GEN_BIN_OP(And, Float64, Bool)
GEN_BIN_OP(And, Float64, Int64)
GEN_BIN_OP(And, Float64, Float64)
GEN_BIN_OP(Or, Bool, Bool)
GEN_BIN_OP(Or, Bool, Int64)
GEN_BIN_OP(Or, Bool, Float64)
GEN_BIN_OP(Or, Int64, Bool)
GEN_BIN_OP(Or, Int64, Int64)
GEN_BIN_OP(Or, Int64, Float64)
GEN_BIN_OP(Or, Float64, Bool)
GEN_BIN_OP(Or, Float64, Int64)
GEN_BIN_OP(Or, Float64, Float64)
GEN_BIN_OP(Xor, Bool, Bool)
GEN_BIN_OP(Xor, Bool, Int64)
GEN_BIN_OP(Xor, Bool, Float64)
GEN_BIN_OP(Xor, Int64, Bool)
GEN_BIN_OP(Xor, Int64, Int64)
GEN_BIN_OP(Xor, Int64, Float64)
GEN_BIN_OP(Xor, Float64, Bool)
GEN_BIN_OP(Xor, Float64, Int64)
GEN_BIN_OP(Xor, Float64, Float64)

GEN_UNI_OP(Not, Bool)
GEN_UNI_OP(Not, Int64)
GEN_UNI_OP(Not, Float64)
GEN_UNI_OP(Bool, Bool)
GEN_UNI_OP(Bool, Int64)
GEN_UNI_OP(Bool, Float64)
GEN_UNI_OP(Abs, Bool)
GEN_UNI_OP(Abs, Int64)
GEN_UNI_OP(Abs, Float64)
GEN_UNI_OP(Int64, Bool)
GEN_UNI_OP(Int64, Int64)
GEN_UNI_OP(Int64, Float64)
GEN_UNI_OP(Ceil, Bool)
GEN_UNI_OP(Ceil, Int64)
GEN_UNI_OP(Ceil, Float64)
GEN_UNI_OP(Rint, Bool)
GEN_UNI_OP(Rint, Int64)
GEN_UNI_OP(Rint, Float64)
GEN_UNI_OP(Floor, Bool)
GEN_UNI_OP(Floor, Int64)
GEN_UNI_OP(Floor, Float64)
GEN_UNI_OP(Exp, Bool)
GEN_UNI_OP(Exp, Int64)
GEN_UNI_OP(Exp, Float64)
GEN_UNI_OP(Expm1, Bool)
GEN_UNI_OP(Expm1, Int64)
GEN_UNI_OP(Expm1, Float64)
GEN_UNI_OP(Log, Bool)
GEN_UNI_OP(Log, Int64)
GEN_UNI_OP(Log, Float64)
GEN_UNI_OP(Log1p, Bool)
GEN_UNI_OP(Log1p, Int64)
GEN_UNI_OP(Log1p, Float64)
GEN_UNI_OP(Log2, Bool)
GEN_UNI_OP(Log2, Int64)
GEN_UNI_OP(Log2, Float64)
GEN_UNI_OP(Log10, Bool)
GEN_UNI_OP(Log10, Int64)
GEN_UNI_OP(Log10, Float64)
GEN_UNI_OP(Sqrt, Bool)
GEN_UNI_OP(Sqrt, Int64)
GEN_UNI_OP(Sqrt, Float64)
GEN_UNI_OP(Asin, Bool)
GEN_UNI_OP(Asin, Int64)
GEN_UNI_OP(Asin, Float64)
GEN_UNI_OP(Acos, Bool)
GEN_UNI_OP(Acos, Int64)
GEN_UNI_OP(Acos, Float64)
GEN_UNI_OP(Atan, Bool)
GEN_UNI_OP(Atan, Int64)
GEN_UNI_OP(Atan, Float64)
GEN_UNI_OP(Asinh, Bool)
GEN_UNI_OP(Asinh, Int64)
GEN_UNI_OP(Asinh, Float64)
GEN_UNI_OP(Acosh, Bool)
GEN_UNI_OP(Acosh, Int64)
GEN_UNI_OP(Acosh, Float64)
GEN_UNI_OP(Atanh, Bool)
GEN_UNI_OP(Atanh, Int64)
GEN_UNI_OP(Atanh, Float64)
GEN_UNI_OP(Sin, Bool)
GEN_UNI_OP(Sin, Int64)
GEN_UNI_OP(Sin, Float64)
GEN_UNI_OP(Cos, Bool)
GEN_UNI_OP(Cos, Int64)
GEN_UNI_OP(Cos, Float64)
GEN_UNI_OP(Tan, Bool)
GEN_UNI_OP(Tan, Int64)
GEN_UNI_OP(Tan, Float64)
GEN_UNI_OP(Sinh, Bool)
GEN_UNI_OP(Sinh, Int64)
GEN_UNI_OP(Sinh, Float64)
GEN_UNI_OP(Cosh, Bool)
GEN_UNI_OP(Cosh, Int64)
GEN_UNI_OP(Cosh, Float64)
GEN_UNI_OP(Tanh, Bool)
GEN_UNI_OP(Tanh, Int64)
GEN_UNI_OP(Tanh, Float64)

GEN_SELECT_OP(Bool, Bool)
GEN_SELECT_OP(Bool, Int64)
GEN_SELECT_OP(Bool, Float64)
GEN_SELECT_OP(Int64, Bool)
GEN_SELECT_OP(Int64, Int64)
GEN_SELECT_OP(Int64, Float64)
GEN_SELECT_OP(Float64, Bool)
GEN_SELECT_OP(Float64, Int64)
GEN_SELECT_OP(Float64, Float64)
