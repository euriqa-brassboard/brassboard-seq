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

#ifndef BRASSBOARD_SEQ_SRC_ACTION_H
#define BRASSBOARD_SEQ_SRC_ACTION_H

#include "utils.h"

namespace brassboard_seq::action {

struct Action {
    py::ref<> value;
    py::ref<> cond;
    py::ref<> kws;
    bool is_pulse;
    bool exact_time;
    bool cond_val;
    int aid;
    int tid;
    int end_tid;
    py::ptr<> length;
    py::ref<> end_val;

    Action(py::ptr<> value, py::ptr<> cond,
           bool is_pulse, bool exact_time, py::dict_ref &&kws, int aid)
        : value(value.ref()),
          cond(cond.ref()),
          kws(std::move(kws)),
          is_pulse(is_pulse),
          exact_time(exact_time),
          cond_val(false),
          aid(aid)
    {
    }

    py::str_ref py_str();
};

using ActionAllocator = PermAllocator<Action,146>;

}

#endif
