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
#include "src/rfsoc.h"
#include "src/rtprop.h"
#include "src/rtval.h"
#include "src/seq.h"

using namespace brassboard_seq;

static PyModuleDef _utils_module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "brassboard_seq._utils",
    .m_size = -1,
};

PY_MODINIT(_utils, _utils_module)
{
    // event_time
    m.add_type(&event_time::TimeManager::Type);
    m.add_type(&event_time::EventTime::Type);
    m.add_objref("event_time_time_scale",
                 py::new_cfunc(&event_time::time_scale_method,
                               nullptr, "brassboard_seq.event_time"_py));

    // rfsoc
    m.add_type(&rfsoc::JaqalInst_v1_Type);
    m.add_type(&rfsoc::Jaqal_v1_Type);
    m.add_type(&rfsoc::JaqalChannelGen_v1_Type);
    m.add_type(&rfsoc::JaqalInst_v1_3_Type);
    m.add_type(&rfsoc::Jaqal_v1_3_Type);

    // rtprop
    m.add_type(&rtprop::CompositeRTProp_Type);
    m.add_type(&rtprop::RTProp_Type);

    // rtval
    m.add_type(&rtval::RuntimeValue::Type);
    m.add_type(&rtval::ExternCallback::Type);
    m.add_objref("rtval_get_value", py::new_cfunc(&rtval::get_value_method,
                                                  nullptr, "brassboard_seq.rtval"_py));
    m.add_objref("rtval_inv", py::new_cfunc(&rtval::inv_method,
                                            nullptr, "brassboard_seq.rtval"_py));
    m.add_objref("rtval_convert_bool",
                 py::new_cfunc(&rtval::convert_bool_method,
                               nullptr, "brassboard_seq.rtval"_py));
    m.add_objref("rtval_ifelse", py::new_cfunc(&rtval::ifelse_method,
                                               nullptr, "brassboard_seq.rtval"_py));
    m.add_objref("rtval_same_value", py::new_cfunc(&rtval::same_value_method,
                                                   nullptr, "brassboard_seq.rtval"_py));

    // seq
    m.add_type(&seq::Seq::Type);
}
