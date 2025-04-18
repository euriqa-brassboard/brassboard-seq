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

#include "scan.h"

namespace brassboard_seq::scan {

// Check if the struct field reference path is overwritten in `obj`.
// Overwrite happens if the field itself exists or a parent of the field
// is overwritten to something that's not scalar struct.
static inline bool check_field(PyObject *d, PyObject *path)
{
    for (auto [_, name]: py::tuple_iter(path)) {
        auto vp = py::dict(d).try_get(name);
        if (!vp)
            return false;
        if (!vp.typeis<py::dict>())
            return true;
        d = vp;
    }
    return true;
}

}
