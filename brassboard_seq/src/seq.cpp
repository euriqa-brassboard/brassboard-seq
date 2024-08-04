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

#include <vector>

namespace brassboard_seq::seq {

static void type_add_method(PyTypeObject *type, PyMethodDef *meth)
{
    py_object descr(PyDescr_NewMethod(type, meth));
    if (!descr || PyDict_SetItemString(type->tp_dict, meth->ml_name, descr) == -1)
        throw 0;
    return;
}

static PyObject *RaiseArgtupleInvalid(const char* func_name, bool exact,
                                      Py_ssize_t num_min, Py_ssize_t num_max,
                                      Py_ssize_t num_found)
{
    Py_ssize_t num_expected;
    const char *more_or_less;
    if (num_found < num_min) {
        num_expected = num_min;
        more_or_less = "at least";
    } else {
        num_expected = num_max;
        more_or_less = "at most";
    }
    if (exact)
        more_or_less = "exactly";
    PyErr_Format(PyExc_TypeError,
                 "%.200s() takes %.8s %zd positional argument%.1s (%zd given)",
                 func_name, more_or_less, num_expected,
                 (num_expected == 1) ? "" : "s", num_found);
    return nullptr;
}

struct SeqVTable {
    int (*timestep_set)(PyObject *self, PyObject *chn, PyObject *value,
                        PyObject *cond, int is_pulse, int exact_time, PyObject *kws);
    int (*subseq_set)(PyObject *self, PyObject *chn, PyObject *value,
                      PyObject *cond, int exact_time, PyObject *kws);
    PyObject *(*combine_cond)(PyObject *cond1, PyObject *new_cond);
};
static SeqVTable seq_vtable;

struct seq_set_params {
    PyObject *chn;
    PyObject *value;
    bool exact_time{false};
    PyObject *cond{Py_True};
    py_object<PyObject> kws;

    PyObject *kwargs() const
    {
        return kws ? kws.get() : Py_None;
    }

    seq_set_params(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames,
                   bool is_pulse)
    {
        if (nargs != 2) {
            RaiseArgtupleInvalid(is_pulse ? "pulse" : "set", true, 2, 2, nargs);
            throw 0;
        }
        chn = args[0];
        value = args[1];

        if (kwnames) {
            auto kwvalues = args + nargs;
            int nkws = (int)PyTuple_GET_SIZE(kwnames);
            for (int i = 0; i < nkws; i++) {
                auto name = PyTuple_GET_ITEM(kwnames, i);
                auto value = kwvalues[i];
                if (PyUnicode_CompareWithASCIIString(name, "cond") == 0) {
                    cond = value;
                }
                else if (PyUnicode_CompareWithASCIIString(name, "exact_time") == 0) {
                    exact_time = get_value_bool(value, (uintptr_t)-1);
                }
                else {
                    if (!kws) {
                        kws.reset(PyDict_New());
                        if (!kws) {
                            throw 0;
                        }
                    }
                    if (PyDict_SetItem(kws, name, value) == -1) {
                        throw 0;
                    }
                }
            }
        }
    }
};

struct CondCombiner {
    PyObject *cond{nullptr};
    bool needs_free{false};
    CondCombiner(PyObject *cond1, PyObject *cond2)
    {
        if (cond1 == Py_True) {
            cond = cond2;
        }
        else if (cond1 == Py_False) {
            cond = Py_False;
        }
        if (cond2 == Py_True) {
            cond = cond1;
        }
        else if (cond2 == Py_False) {
            cond = Py_False;
        }
        cond = seq_vtable.combine_cond(cond1, cond2);
        if (!cond)
            throw 0;
        needs_free = true;
    }
    ~CondCombiner()
    {
        if (needs_free) {
            Py_DECREF(cond);
        }
    }
};

template<bool is_cond, typename CondSeq>
static inline auto condseq_get_subseq(CondSeq *self)
{
    if constexpr (is_cond) {
        return self->seq;
    }
    else {
        return self;
    }
}

template<bool is_cond, typename CondSeq>
static inline auto condseq_get_cond(CondSeq *self)
{
    if constexpr (is_cond) {
        return self->cond;
    }
    else {
        return self->__pyx_base.cond;
    }
}

template<typename TimeStep, bool is_pulse>
static PyObject *timestep_set(PyObject *self, PyObject *const *args,
                              Py_ssize_t nargs, PyObject *kwnames) try
{
    seq_set_params params(args, nargs, kwnames, is_pulse);
    TimeStep *step = (TimeStep*)self;
    CondCombiner cc(step->__pyx_base.cond, params.cond);
    auto res = seq_vtable.timestep_set(self, params.chn, params.value, cc.cond,
                                       is_pulse, params.exact_time, params.kwargs());
    if (res == -1)
        return nullptr;
    Py_INCREF(self);
    return self;
}
catch (...) {
    return nullptr;
}

template<typename TimeStep>
static inline void
update_timestep(PyTypeObject *ty_timestep, TimeStep*)
{
    static PyMethodDef timestep_set_method = {
        "set", (PyCFunction)(void*)timestep_set<TimeStep,false>,
        METH_FASTCALL|METH_KEYWORDS, 0};
    static PyMethodDef timestep_pulse_method = {
        "pulse", (PyCFunction)(void*)timestep_set<TimeStep,true>,
        METH_FASTCALL|METH_KEYWORDS, 0};
    type_add_method(ty_timestep, &timestep_set_method);
    type_add_method(ty_timestep, &timestep_pulse_method);
    PyType_Modified(ty_timestep);
}

template<typename CondSeq, bool is_cond>
static PyObject *condseq_set(PyObject *py_self, PyObject *const *args,
                             Py_ssize_t nargs, PyObject *kwnames) try
{
    seq_set_params params(args, nargs, kwnames, false);
    auto self = (CondSeq*)py_self;
    auto subseq = condseq_get_subseq<is_cond>(self);
    auto cond = condseq_get_cond<is_cond>(self);
    CondCombiner cc(cond, params.cond);
    auto res = seq_vtable.subseq_set((PyObject*)subseq, params.chn, params.value,
                                     cc.cond, params.exact_time, params.kwargs());
    if (res == -1)
        return nullptr;
    Py_INCREF(py_self);
    return py_self;
}
catch (...) {
    return nullptr;
}

template<typename SubSeq>
static inline void
update_subseq(PyTypeObject *ty_subseq, SubSeq*)
{
    static PyMethodDef subseq_set_method = {
        "set", (PyCFunction)(void*)condseq_set<SubSeq,false>,
        METH_FASTCALL|METH_KEYWORDS, 0};
    type_add_method(ty_subseq, &subseq_set_method);
    PyType_Modified(ty_subseq);
}

template<typename ConditionalWrapper>
static inline void
update_conditional(PyTypeObject *ty_conditional, ConditionalWrapper*)
{
    static PyMethodDef conditional_set_method = {
        "set", (PyCFunction)(void*)condseq_set<ConditionalWrapper,true>,
        METH_FASTCALL|METH_KEYWORDS, 0};
    type_add_method(ty_conditional, &conditional_set_method);
    PyType_Modified(ty_conditional);
}

}
