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

#ifndef BRASSBOARD_SEQ_SRC_SCAN_H
#define BRASSBOARD_SEQ_SRC_SCAN_H

#include "Python.h"

#include "utils.h"

namespace brassboard_seq::scan {

template<typename ParamPack>
static inline __attribute__((returns_nonnull)) ParamPack*
new_param_pack(PyObject *type, PyObject *values, PyObject *visited,
               PyObject *fieldname, ParamPack*)
{
    auto o = pytype_genericalloc(type);
    auto self = (ParamPack*)o;
    self->values = py_newref(values);
    self->visited = py_newref(visited);
    self->fieldname = py_newref(fieldname);
    return self;
}

}

#endif
