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

#include "action.h"

namespace brassboard_seq::action {

py::str_ref Action::py_str()
{
    py::stringio io;
    io.write_ascii(is_pulse ? "Pulse(" : "Set(");
    io.write_str(value);
    if (cond != Py_True) {
        io.write_ascii(", cond=");
        io.write_str(cond);
    }
    if (exact_time)
        io.write_ascii(", exact_time=True");
    if (kws) {
        for (auto [name, val]: py::dict_iter(kws)) {
            io.write_ascii(", ");
            io.write(name);
            io.write_ascii("=");
            io.write_str(val);
        }
    }
    io.write_ascii(")");
    return io.getvalue();
}

}
