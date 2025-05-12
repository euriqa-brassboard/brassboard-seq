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

#ifndef BRASSBOARD_SEQ_TEST_UTILS_H
#define BRASSBOARD_SEQ_TEST_UTILS_H

#include "src/utils.h"

using namespace brassboard_seq;

template<str_literal name> static auto new_object()
{
    static py::ptr type = py::ptr(&PyType_Type)(py::new_str(name), py::new_tuple(),
                                                py::new_dict()).rel();
    return type();
}

static auto value_pair_list(auto &values)
{
    return py::new_nlist(values.size(), [&] (int i) { return py::ptr(values[i].first); });
}

#endif
