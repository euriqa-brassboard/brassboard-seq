# cython: language_level=3

# Copyright (c) 2024 - 2025 Yichao Yu <yyc1992@gmail.com>

# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 3.0 of the License, or (at your option) any later version.

# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public
# License along with this library. If not,
# see <http://www.gnu.org/licenses/>.

# Do not use relative import since it messes up cython file name tracking

from brassboard_seq._utils import CompositeRTProp, RTProp
# Manually set the field since I can't make cython automatically do this
# without also declaring the c struct again...
globals()['RuntimeValue'] = RuntimeValue

def get_value(v, unsigned age):
    if is_rtval(v):
        rt_eval_cache(<RuntimeValue>v, age)
        return rtval_cache(<RuntimeValue>v).to_py()
    return v

def inv(v, /):
    if type(v) is bool:
        return v is False
    cdef RuntimeValue _v
    if is_rtval(v):
        _v = <RuntimeValue>v
        if _v.type_ == ValueType.Not:
            return rt_convert_bool(_v.arg0)
        return new_expr1(ValueType.Not, _v)
    return not v

def convert_bool(_v):
    if is_rtval(_v):
        return rt_convert_bool(<RuntimeValue>_v)
    return bool(_v)

def ifelse(b, v1, v2):
    if rt_same_value(v1, v2):
        return v1
    if is_rtval(b):
        return new_select(<RuntimeValue>b, v1, v2)
    return v1 if b else v2

def same_value(v1, v2):
    return rt_same_value(v1, v2)
