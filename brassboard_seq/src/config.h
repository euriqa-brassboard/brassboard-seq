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

#ifndef BRASSBOARD_SEQ_SRC_CONFIG_H
#define BRASSBOARD_SEQ_SRC_CONFIG_H

#include "utils.h"

namespace brassboard_seq::config {

struct Config : PyObject {
    py::dict_ref channel_alias;
    py::dict_ref alias_cache;
    py::set_ref supported_prefix;

    PyObject *translate_channel(PyObject *name);

    static PyTypeObject Type;
private:
    py::tuple_ref _translate_channel(py::tuple path);
};

}

#endif
