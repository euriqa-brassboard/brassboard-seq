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
#include "src/rtprop.h"
#include "src/rtval.h"
#include "src/config.h"

static PyModuleDef _utils_module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "brassboard_seq._utils",
    .m_doc = "Backing module for brassboard_seq",
    .m_size = -1,
};

PyMODINIT_FUNC
PyInit__utils(void)
{
    using namespace brassboard_seq;
    return py_catch_error([&] {
        rtval::init();
        rtprop::init();
        config::init();
        py_object m(throw_if_not(PyModule_Create(&_utils_module)));
        throw_if(PyModule_AddObjectRef(m, "RuntimeValue",
                                       (PyObject*)&rtval::RuntimeValue_Type) < 0);
        throw_if(PyModule_AddObjectRef(m, "ExternCallback",
                                       (PyObject*)&rtval::ExternCallback_Type) < 0);
        throw_if(PyModule_AddObjectRef(m, "Config",
                                       (PyObject*)&config::Config_Type) < 0);
        throw_if(PyModule_AddObjectRef(m, "CompositeRTProp",
                                       (PyObject*)&rtprop::CompositeRTProp_Type) < 0);
        throw_if(PyModule_AddObjectRef(m, "RTProp",
                                       (PyObject*)&rtprop::RTProp_Type) < 0);
        return m.release();
    });
}
