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

#ifndef BRASSBOARD_SEQ_SRC_RTVAL_H
#define BRASSBOARD_SEQ_SRC_RTVAL_H

#include "lib_rtval.h"

namespace brassboard_seq::rtval {

template<typename RuntimeValue>
static inline __attribute__((returns_nonnull)) RuntimeValue*
rt_convert_bool(RuntimeValue *v)
{
    if (v->type_ == Int64)
        v = v->arg0;
    if (v->datatype == DataType::Bool)
        return py_newref(v);
    return new_expr1(Bool, v);
}

template<typename RuntimeValue>
static inline __attribute__((returns_nonnull)) RuntimeValue*
rt_round_int64(RuntimeValue *v)
{
    if (v->type_ == Int64)
        return py_newref(v);
    return new_expr1(Int64, v);
}

}

#endif
