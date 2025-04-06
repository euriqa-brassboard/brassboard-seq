#define PY_SSIZE_T_CLEAN

#include <Python.h>

#include "src/utils.h"
#include "src/lib_rtval.h"

static PyModuleDef _utils_module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "brassboard_seq._utils",
    .m_doc = "Backing module for brassboard_seq",
    .m_size = -1,
};

PyMODINIT_FUNC
PyInit__utils(void)
{
    using namespace brassboard_seq::rtval;

    if (PyType_Ready(&RuntimeValue_Type) < 0)
        return nullptr;

    PyObject *m = PyModule_Create(&_utils_module);
    if (!m)
        return nullptr;

    if (PyModule_AddObjectRef(m, "RuntimeValue", (PyObject*)&RuntimeValue_Type) < 0) {
        Py_DECREF(m);
        return nullptr;
    }

    return m;
}
