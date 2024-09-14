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

#include "Python.h"

#include "utils.h"

namespace brassboard_seq::scan {

void merge_dict_into(PyObject *tgt, PyObject *src, bool ovr)
{
    PyObject *key;
    PyObject *value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(src, &pos, &key, &value)) {
        auto oldv = PyDict_GetItemWithError(tgt, key);
        if (oldv) {
            bool is_dict = PyDict_Check(value);
            bool was_dict = PyDict_Check(oldv);
            if (was_dict && !is_dict) {
                PyErr_Format(PyExc_TypeError,
                             "Cannot override parameter pack as value");
                throw 0;
            }
            else if (!was_dict && is_dict) {
                PyErr_Format(PyExc_TypeError,
                             "Cannot override value as parameter pack");
                throw 0;
            }
            else if (is_dict) {
                merge_dict_into(oldv, value, ovr);
            }
            else if (ovr && PyDict_SetItem(tgt, key, value) < 0) {
                throw 0;
            }
        }
        else if (PyErr_Occurred()) {
            throw 0;
        }
        else {
            py_object copied(pydict_deepcopy(value));
            if (PyDict_SetItem(tgt, key, copied.get()) < 0) {
                throw 0;
            }
        }
    }
}


}
