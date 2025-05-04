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

#include "src/utils.h"
#include "src/event_time.h"

using namespace brassboard_seq;

static PyModuleDef event_time_module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "brassboard_seq.event_time",
    .m_size = -1,
    .m_methods = event_time::methods,
};

PY_MODINIT(event_time, event_time_module)
{
    m.add_type(&event_time::EventTime::Type);
}
