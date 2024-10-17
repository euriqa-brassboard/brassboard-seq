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

#ifndef BRASSBOARD_SEQ_SRC_ACTION_H
#define BRASSBOARD_SEQ_SRC_ACTION_H

namespace brassboard_seq::action {

struct ActionData {
    bool is_pulse;
    bool exact_time;
    bool cond_val;
};

template<typename Action>
static inline __attribute__((returns_nonnull)) Action*
new_action(PyObject *ActionType, PyObject *value, PyObject *cond,
           bool is_pulse, bool exact_time, PyObject *kws, int aid, Action*)
{
    auto o = pytype_genericalloc(ActionType);
    auto p = (Action*)o;
    p->data.is_pulse = is_pulse;
    p->data.exact_time = exact_time;
    p->value = py_newref(value);
    p->cond = py_newref(cond);
    p->kws = py_newref(kws);
    p->aid = aid;

    p->length = py_newref(Py_None);
    p->end_val = py_newref(Py_None);
    return p;
}

}

#endif
