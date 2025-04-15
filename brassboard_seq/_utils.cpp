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

#include "src/config.h"
#include "src/event_time.h"
#include "src/rtprop.h"
#include "src/rtval.h"
#include "src/scan.h"
#include "src/yaml.h"

using namespace brassboard_seq;

static PyModuleDef _utils_module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "brassboard_seq._utils",
    .m_doc = "Backing module for brassboard_seq",
    .m_size = -1,
};

#if PY_VERSION_HEX >= 0x030a0000
static void pymodule_addobjectref(py::mod m, const char *name, PyObject *value)
{
    throw_if(PyModule_AddObjectRef((PyObject*)m, name, value) < 0);
}
#else
static void pymodule_addobjectref(py::mod m, const char *name, PyObject *value)
{
    py::ref v(py::newref(value));
    throw_if(PyModule_AddObject((PyObject*)m, name, value.get()) < 0);
    v.rel();
}
#endif

PyMODINIT_FUNC
PyInit__utils(void)
{
    return cxx_catch([&] {
        auto yaml_sprint = py::new_cfunc(&yaml::sprint_method, nullptr,
                                         "brassboard_seq.yaml"_py);
        auto parampack_get_visited = py::new_cfunc(&scan::parampack_get_visited_method,
                                                   nullptr, "brassboard_seq.scan"_py);
        auto parampack_get_param = py::new_cfunc(&scan::parampack_get_param_method,
                                                 nullptr, "brassboard_seq.scan"_py);
        auto m = py::new_module(&_utils_module);
        pymodule_addobjectref(m, "RuntimeValue",
                              (PyObject*)&rtval::RuntimeValue::Type);
        pymodule_addobjectref(m, "ExternCallback",
                              (PyObject*)&rtval::ExternCallback::Type);
        pymodule_addobjectref(m, "TimeManager",
                              (PyObject*)&event_time::TimeManager::Type);
        pymodule_addobjectref(m, "EventTime",
                              (PyObject*)&event_time::EventTime::Type);
        pymodule_addobjectref(m, "Config", (PyObject*)&config::Config::Type);
        pymodule_addobjectref(m, "CompositeRTProp",
                              (PyObject*)&rtprop::CompositeRTProp_Type);
        pymodule_addobjectref(m, "RTProp", (PyObject*)&rtprop::RTProp_Type);
        pymodule_addobjectref(m, "ParamPack", (PyObject*)&scan::ParamPack::Type);
        pymodule_addobjectref(m, "parampack_get_visited", parampack_get_visited.get());
        pymodule_addobjectref(m, "parampack_get_param", parampack_get_param.get());
        pymodule_addobjectref(m, "yaml_sprint", yaml_sprint.get());
        return m;
    });
}
