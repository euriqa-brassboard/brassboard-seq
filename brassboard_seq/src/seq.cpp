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

static PyObject *__pyx_f_14brassboard_seq_3seq_combine_cond(PyObject*, PyObject*);

namespace brassboard_seq::seq {

static void type_add_method(PyTypeObject *type, PyMethodDef *meth)
{
    py_object descr(throw_if_not(PyDescr_NewMethod(type, meth)));
    throw_if_not(PyDict_SetItemString(type->tp_dict, meth->ml_name, descr) == 0);
}

static void raise_too_few_args(const char* func_name, bool exact,
                               Py_ssize_t num_min, Py_ssize_t num_found)
{
    const char *more_or_less = exact ? "exactly" : "at least";
    PyErr_Format(PyExc_TypeError,
                 "%.200s() takes %.8s %zd positional argument%.1s (%zd given)",
                 func_name, more_or_less, num_min,
                 (num_min == 1) ? "" : "s", num_found);
    throw 0;
}

static PyTypeObject *event_time_type;

struct seq_set_params {
    PyObject *chn;
    PyObject *value;
    bool exact_time{false};
    PyObject *cond{Py_True};
    py_object kws;

    PyObject *kwargs() const
    {
        return kws ? kws.get() : Py_None;
    }

    seq_set_params(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames,
                   bool is_pulse)
    {
        if (nargs != 2)
            raise_too_few_args((is_pulse ? "pulse" : "set"), true, 2, nargs);
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
                    if (!kws)
                        kws.reset(throw_if_not(PyDict_New()));
                    throw_if_not(PyDict_SetItem(kws, name, value) == 0);
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
        cond = throw_if_not(__pyx_f_14brassboard_seq_3seq_combine_cond(cond1, cond2));
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

enum class AddStepType {
    Step,
    Background,
    Floating,
    At,
};

const char *add_step_name(AddStepType type)
{
    if (type == AddStepType::Background)
        return "add_background";
    if (type == AddStepType::Floating)
        return "add_floating";
    if (type == AddStepType::At)
        return "add_at";
    assert(type == AddStepType::Step);
    return "add_step";
}

static auto empty_tuple = PyTuple_New(0);

template<typename CondSeq, typename TimeSeq, bool is_cond, AddStepType type>
static PyObject *add_step_real(PyObject *py_self, PyObject *const *args,
                               Py_ssize_t nargs, PyObject *kwnames) try
{
    auto self = (CondSeq*)py_self;
    auto subseq = condseq_get_subseq<is_cond>(self);
    auto cond = condseq_get_cond<is_cond>(self);
    auto nargs_min = type == AddStepType::At ? 2 : 1;
    if (nargs < nargs_min)
        raise_too_few_args(add_step_name(type), false, nargs_min, nargs);

    auto first_arg = args[nargs_min - 1];
    using EventTime = std::remove_reference_t<decltype(*subseq->__pyx_base.end_time)>;
    py_object start_time;
    if (type == AddStepType::Background) {
        start_time.reset(py_newref((PyObject*)subseq->__pyx_base.end_time));
    }
    else if (type == AddStepType::Floating) {
        start_time.reset(
            throw_if_not(
                __pyx_f_14brassboard_seq_3seq_new_floating_time(&subseq->__pyx_base,
                                                                cond)));
    }
    else if (type == AddStepType::At) {
        if (args[0] != Py_None && Py_TYPE(args[0]) != event_time_type)
            return PyErr_Format(PyExc_TypeError,
                                "Argument 'tp' has incorrect type (expected EventTime, "
                                "got %.200s)", Py_TYPE(args[0])->tp_name);
        start_time.reset(py_newref(args[0]));
    }
    else {
        assert(type == AddStepType::Step);
        start_time.reset(py_newref((PyObject*)subseq->__pyx_base.end_time));
    }

    auto tuple_nargs = nargs - nargs_min;
    auto get_args_tuple = [&] {
        if (tuple_nargs == 0)
            return py_newref(empty_tuple);
        auto res = throw_if_not(PyTuple_New(tuple_nargs));
        auto *tuple_args = args + nargs_min;
        for (auto i = 0; i < tuple_nargs; i++)
            PyTuple_SET_ITEM(res, i, py_newref(tuple_args[i]));
        return res;
    };

    py_object kws;
    if (kwnames) {
        kws.reset(throw_if_not(PyDict_New()));
        auto kwvalues = args + nargs;
        int nkws = (int)PyTuple_GET_SIZE(kwnames);
        for (int i = 0; i < nkws; i++) {
            throw_if_not(PyDict_SetItem(kws, PyTuple_GET_ITEM(kwnames, i),
                                        kwvalues[i]) == 0);
        }
    }

    PyObject *res;
    if (Py_TYPE(first_arg)->tp_call) {
        py_object arg_tuple(get_args_tuple());
        res = (PyObject*)throw_if_not(
            __pyx_f_14brassboard_seq_3seq_add_custom_step(
                subseq, cond, (EventTime*)start_time.get(),
                first_arg, arg_tuple,
                kws ? kws.get() : Py_None));
    }
    else if (kws) {
        py_object arg_tuple(get_args_tuple());
        return PyErr_Format(PyExc_ValueError,
                            "Unexpected arguments when creating new time step, %S, %S.",
                            arg_tuple.get(), kws.get());
    }
    else if (tuple_nargs == 0) {
        res = (PyObject*)throw_if_not(
            __pyx_f_14brassboard_seq_3seq_add_time_step(
                subseq, cond, (EventTime*)start_time.get(), first_arg));
    }
    else {
        py_object arg_tuple(get_args_tuple());
        return PyErr_Format(PyExc_ValueError,
                            "Unexpected arguments when creating new time step, %S.",
                            arg_tuple.get());
    }
    if (type == AddStepType::Step) {
        auto new_seq = (TimeSeq*)res;
        auto prev_time = subseq->__pyx_base.end_time;
        subseq->__pyx_base.end_time = py_newref(new_seq->end_time);
        Py_DECREF(prev_time);
    }
    return res;
}
catch (...) {
    return nullptr;
}

template<typename CondSeq, bool is_cond, bool is_step=false, bool is_pulse=false>
static PyObject *condseq_set(PyObject *py_self, PyObject *const *args,
                             Py_ssize_t nargs, PyObject *kwnames) try
{
    seq_set_params params(args, nargs, kwnames, is_pulse);
    auto self = (CondSeq*)py_self;
    auto subseq = condseq_get_subseq<is_cond>(self);
    auto cond = condseq_get_cond<is_cond>(self);
    CondCombiner cc(cond, params.cond);
    if constexpr (is_step)
        throw_if_not(
            __pyx_f_14brassboard_seq_3seq_timestep_set(
                subseq, params.chn, params.value, cc.cond, is_pulse, params.exact_time,
                params.kwargs()) == 0);
    else
        throw_if_not(
            __pyx_f_14brassboard_seq_3seq_subseq_set(
                subseq, params.chn, params.value,
                cc.cond, params.exact_time, params.kwargs()) == 0);
    return py_newref(py_self);
}
catch (...) {
    return nullptr;
}

template<typename TimeStep>
static inline void
update_timestep(PyTypeObject *ty_timestep, TimeStep*)
{
    static PyMethodDef timestep_set_method = {
        "set", (PyCFunction)(void*)condseq_set<TimeStep,false,true,false>,
        METH_FASTCALL|METH_KEYWORDS, 0};
    static PyMethodDef timestep_pulse_method = {
        "pulse", (PyCFunction)(void*)condseq_set<TimeStep,false,true,true>,
        METH_FASTCALL|METH_KEYWORDS, 0};
    type_add_method(ty_timestep, &timestep_set_method);
    type_add_method(ty_timestep, &timestep_pulse_method);
    PyType_Modified(ty_timestep);
}

template<typename SubSeq, typename TimeSeq>
static inline void
update_subseq(PyTypeObject *ty_subseq, SubSeq*, TimeSeq*)
{
    static PyMethodDef subseq_set_method = {
        "set", (PyCFunction)(void*)condseq_set<SubSeq,false>,
        METH_FASTCALL|METH_KEYWORDS, 0};
    static PyMethodDef subseq_add_step_method = {
        "add_step",
        (PyCFunction)(void*)add_step_real<SubSeq,TimeSeq,false,AddStepType::Step>,
        METH_FASTCALL|METH_KEYWORDS, 0};
    static PyMethodDef subseq_add_background_method = {
        "add_background",
        (PyCFunction)(void*)add_step_real<SubSeq,TimeSeq,false,AddStepType::Background>,
        METH_FASTCALL|METH_KEYWORDS, 0};
    static PyMethodDef subseq_add_floating_method = {
        "add_floating",
        (PyCFunction)(void*)add_step_real<SubSeq,TimeSeq,false,AddStepType::Floating>,
        METH_FASTCALL|METH_KEYWORDS, 0};
    static PyMethodDef subseq_add_at_method = {
        "add_at",
        (PyCFunction)(void*)add_step_real<SubSeq,TimeSeq,false,AddStepType::At>,
        METH_FASTCALL|METH_KEYWORDS, 0};
    type_add_method(ty_subseq, &subseq_set_method);
    type_add_method(ty_subseq, &subseq_add_step_method);
    type_add_method(ty_subseq, &subseq_add_background_method);
    type_add_method(ty_subseq, &subseq_add_floating_method);
    type_add_method(ty_subseq, &subseq_add_at_method);
    PyType_Modified(ty_subseq);
}

template<typename ConditionalWrapper, typename TimeSeq>
static inline void
update_conditional(PyTypeObject *ty_conditional, ConditionalWrapper*, TimeSeq*)
{
    static PyMethodDef conditional_set_method = {
        "set", (PyCFunction)(void*)condseq_set<ConditionalWrapper,true>,
        METH_FASTCALL|METH_KEYWORDS, 0};
    static PyMethodDef conditional_add_step_method = {
        "add_step",
        (PyCFunction)(void*)add_step_real<ConditionalWrapper,TimeSeq,true,AddStepType::Step>,
        METH_FASTCALL|METH_KEYWORDS, 0};
    static PyMethodDef conditional_add_background_method = {
        "add_background",
        (PyCFunction)(void*)add_step_real<ConditionalWrapper,TimeSeq,true,AddStepType::Background>,
        METH_FASTCALL|METH_KEYWORDS, 0};
    static PyMethodDef conditional_add_floating_method = {
        "add_floating",
        (PyCFunction)(void*)add_step_real<ConditionalWrapper,TimeSeq,true,AddStepType::Floating>,
        METH_FASTCALL|METH_KEYWORDS, 0};
    static PyMethodDef conditional_add_at_method = {
        "add_at",
        (PyCFunction)(void*)add_step_real<ConditionalWrapper,TimeSeq,true,AddStepType::At>,
        METH_FASTCALL|METH_KEYWORDS, 0};
    type_add_method(ty_conditional, &conditional_set_method);
    type_add_method(ty_conditional, &conditional_add_step_method);
    type_add_method(ty_conditional, &conditional_add_background_method);
    type_add_method(ty_conditional, &conditional_add_floating_method);
    type_add_method(ty_conditional, &conditional_add_at_method);
    PyType_Modified(ty_conditional);
}

}
