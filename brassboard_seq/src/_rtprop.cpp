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

static TagVal rtprop_callback_func(auto *self, unsigned age)
{
    py_object v(throw_if_not(PyObject_GetAttr(self->obj, self->fieldname)));
    if (!is_rtval(v))
        return TagVal::from_py(v);
    auto rv = (RuntimeValue*)v.get();
    if (rv->type_ == ExternAge && rv->cb_arg2 == (PyObject*)self)
        py_throw_format(PyExc_ValueError, "RT property have not been assigned.");
    rt_eval_cache(rv, age);
    return rtval_cache(rv);
}

}
