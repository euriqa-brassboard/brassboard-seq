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

#include "event_time.h"

#include "Python.h"

#include "assert.h"

#include <array>
#include <charconv>

namespace brassboard_seq::event_time {

PyObject *_str_time(long long t)
{
    assert(time_scale == 1e12);
    assert(t >= 0);

    std::array<char, 32> str;

    auto [ptr, ec] = std::to_chars(str.data(), str.data() + str.size(), t);
    if (ec != std::errc()) {
        PyErr_Format(PyExc_RuntimeError, "%s",
                     std::make_error_code(ec).message().c_str());
        return nullptr;
    }
    *ptr = 0;

    auto s = time_scale;
    auto ms = s / 1000;
    auto us = ms / 1000;
    auto ns = us / 1000;

    int dec = 0;
    const char *unit = "ps";
    if (t >= s / 10 * 3) {
        dec = 12;
        unit = "s";
    }
    else if (t >= ms / 10 * 3) {
        dec = 9;
        unit = "ms";
    }
    else if (t >= us / 10 * 3) {
        dec = 6;
        unit = "us";
    }
    else if (t >= ns / 10 * 3) {
        dec = 3;
        unit = "ns";
    }
    char *end = ptr;
    assert(end - str.data() >= dec);
    for (int i = 0; i < dec; i++) {
        if (ptr[-1 - i] != '0')
            break;
        end = &ptr[-1 - i];
        *end = 0;
    }
    auto pdec = ptr - dec;
    if (pdec == str.data()) {
        memmove(pdec + 2, pdec, end - pdec);
        pdec[0] = '0';
        pdec[1] = '.';
        end += 2;
    }
    else if (pdec != end) {
        memmove(pdec + 1, pdec, end - pdec);
        pdec[0] = '.';
        end += 1;
    }
    *end = ' ';
    memcpy(&end[1], unit, strlen(unit) + 1);
    return PyUnicode_FromString(str.data());
}

static traverseproc event_time_base_traverse;
static inquiry event_time_base_clear;

template<typename EventTime>
static inline void update_event_time_gc_callback(PyTypeObject *type, EventTime*)
{
    event_time_base_traverse = type->tp_traverse;
    event_time_base_clear = type->tp_clear;
    type->tp_traverse = [] (PyObject *obj, visitproc visit, void *arg) -> int {
        auto t = (EventTime*)obj;
        if (auto rt_offset = t->data.get_rt_offset()) {
            if (auto e = (*visit)(rt_offset, arg)) {
                return e;
            }
        }
        return event_time_base_traverse(obj, visit, arg);
    };
    type->tp_clear = [] (PyObject *obj) -> int {
        auto t = (EventTime*)obj;
        t->data.clear_rt_offset();
        return event_time_base_clear(obj);
    };
}

}
