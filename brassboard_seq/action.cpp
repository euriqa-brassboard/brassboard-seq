/*************************************************************************
 *   Copyright (c) 2025 - 2025 Yichao Yu <yyc1992@gmail.com>             *
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

#define PY_SSIZE_T_CLEAN

#include <Python.h>

#include "src/utils.h"
#include "src/action.h"

using namespace brassboard_seq;

static PyModuleDef action_module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "brassboard_seq.action",
    .m_size = -1,
};

PY_MODINIT(action)
{
    init();
    auto m = py::new_module(&action_module);
    m.add_objref("RampFunction", &action::RampFunction_Type);
    m.add_objref("SeqCubicSpline", &action::SeqCubicSpline::Type);
    m.add_objref("Blackman", &action::Blackman_Type);
    m.add_objref("BlackmanSquare", &action::BlackmanSquare_Type);
    m.add_objref("LinearRamp", &action::LinearRamp_Type);
    return m;
}
