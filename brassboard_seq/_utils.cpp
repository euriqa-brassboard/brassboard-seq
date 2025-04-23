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
#include "src/config.h"
#include "src/event_time.h"
#include "src/rtprop.h"
#include "src/rtval.h"
#include "src/scan.h"
#include "src/seq.h"
#include "src/yaml.h"

using namespace brassboard_seq;

static PyModuleDef _utils_module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "brassboard_seq._utils",
    .m_doc = "Backing module for brassboard_seq",
    .m_size = -1,
};

#if PY_VERSION_HEX >= 0x030a0000
static void pymodule_addobjectref(py::mod m, const char *name, py::ptr<> value)
{
    throw_if(PyModule_AddObjectRef((PyObject*)m, name, value) < 0);
}
#else
static void pymodule_addobjectref(py::mod m, const char *name, py::ptr<> value)
{
    py::ref v(py::newref(value));
    throw_if(PyModule_AddObject((PyObject*)m, name, v.get()) < 0);
    v.rel();
}
#endif

PyMODINIT_FUNC
PyInit__utils(void)
{
    return cxx_catch([&] {
        auto m = py::new_module(&_utils_module);
        // action
        pymodule_addobjectref(m, "RampFunction", &action::RampFunction_Type);
        pymodule_addobjectref(m, "SeqCubicSpline", &action::SeqCubicSpline::Type);
        pymodule_addobjectref(m, "Blackman", &action::Blackman_Type);
        pymodule_addobjectref(m, "BlackmanSquare", &action::BlackmanSquare_Type);
        pymodule_addobjectref(m, "LinearRamp", &action::LinearRamp_Type);

        // config
        pymodule_addobjectref(m, "Config", &config::Config::Type);

        // event_time
        pymodule_addobjectref(m, "TimeManager", &event_time::TimeManager::Type);
        pymodule_addobjectref(m, "EventTime", &event_time::EventTime::Type);

        // rtprop
        pymodule_addobjectref(m, "CompositeRTProp", &rtprop::CompositeRTProp_Type);
        pymodule_addobjectref(m, "RTProp", &rtprop::RTProp_Type);

        // rtval
        pymodule_addobjectref(m, "RuntimeValue", &rtval::RuntimeValue::Type);
        pymodule_addobjectref(m, "ExternCallback", &rtval::ExternCallback::Type);
        pymodule_addobjectref(m, "rtval_get_value",
                              py::new_cfunc(&rtval::get_value_method,
                                            nullptr, "brassboard_seq.rtval"_py));
        pymodule_addobjectref(m, "rtval_inv",
                              py::new_cfunc(&rtval::inv_method,
                                            nullptr, "brassboard_seq.rtval"_py));
        pymodule_addobjectref(m, "rtval_convert_bool",
                              py::new_cfunc(&rtval::convert_bool_method,
                                            nullptr, "brassboard_seq.rtval"_py));
        pymodule_addobjectref(m, "rtval_ifelse",
                              py::new_cfunc(&rtval::ifelse_method,
                                            nullptr, "brassboard_seq.rtval"_py));
        pymodule_addobjectref(m, "rtval_same_value",
                              py::new_cfunc(&rtval::same_value_method,
                                            nullptr, "brassboard_seq.rtval"_py));

        // scan
        pymodule_addobjectref(m, "ParamPack", &scan::ParamPack::Type);
        pymodule_addobjectref(m, "parampack_get_visited",
                              py::new_cfunc(&scan::parampack_get_visited_method,
                                            nullptr, "brassboard_seq.scan"_py));
        pymodule_addobjectref(m, "parampack_get_param",
                              py::new_cfunc(&scan::parampack_get_param_method,
                                            nullptr, "brassboard_seq.scan"_py));
        pymodule_addobjectref(m, "ScanGroup", &scan::ScanGroup_Type);

        // seq
        pymodule_addobjectref(m, "SeqInfo", &seq::SeqInfo::Type);
        pymodule_addobjectref(m, "TimeSeq", &seq::TimeSeq::Type);
        pymodule_addobjectref(m, "TimeStep", &seq::TimeStep::Type);
        pymodule_addobjectref(m, "SubSeq", &seq::SubSeq::Type);
        pymodule_addobjectref(m, "ConditionalWrapper", &seq::ConditionalWrapper::Type);
        pymodule_addobjectref(m, "Seq", &seq::Seq::Type);

        // yaml
        pymodule_addobjectref(m, "yaml_sprint",
                              py::new_cfunc(&yaml::sprint_method, nullptr,
                                            "brassboard_seq.yaml"_py));
        return m;
    });
}
