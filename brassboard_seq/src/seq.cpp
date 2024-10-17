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

#include "event_time.h"

#include <vector>

namespace brassboard_seq::seq {

static PyTypeObject *event_time_type;
static PyTypeObject *runtime_value_type;
static PyTypeObject *timestep_type;
static PyTypeObject *subseq_type;
static PyTypeObject *condwrapper_type;
static PyObject *rt_time_scale;

template<typename RuntimeValue>
static inline std::pair<PyObject*,bool>
_combine_cond(PyObject *cond1, PyObject *new_cond, RuntimeValue*)
{
    if (cond1 == Py_False)
        return { Py_False, false };
    if (Py_TYPE(new_cond) != runtime_value_type) {
        if (get_value_bool(new_cond, (uintptr_t)-1)) {
            return { cond1, false };
        }
        else {
            return { Py_False, false };
        }
    }
    py_object cond2((PyObject*)rtval::rt_convert_bool((PyObject*)runtime_value_type,
                                                      (RuntimeValue*)new_cond));
    if (cond1 == Py_True)
        return { cond2.release(), true };
    assert(Py_TYPE(cond1) == runtime_value_type);
    auto o = pytype_genericalloc(runtime_value_type);
    auto self = (RuntimeValue*)o;
    self->datatype = rtval::DataType::Bool;
    // self->cache_err = rtval::EvalError::NoError;
    // self->cache_val = { .i64_val = 0 };
    self->type_ = rtval::And;
    self->age = (unsigned)-1;
    self->arg0 = (RuntimeValue*)py_newref(cond1);
    self->arg1 = (RuntimeValue*)cond2.release();
    self->cb_arg2 = py_newref(Py_None);
    return { o, true };
}

template<typename RuntimeValue>
static inline __attribute__((returns_nonnull)) PyObject*
combine_cond(PyObject *cond1, PyObject *new_cond, RuntimeValue*)
{
    auto [res, needs_free] = _combine_cond(cond1, new_cond, (RuntimeValue*)nullptr);
    if (!needs_free)
        Py_INCREF(res);
    return res;
}

template<typename TimeManager, typename EventTime, typename RuntimeValue>
static inline __attribute__((returns_nonnull)) EventTime*
new_round_time(TimeManager *self, EventTime *prev, PyObject *offset, PyObject *cond,
               EventTime *wait_for, RuntimeValue*)
{
    if (Py_TYPE(offset) == runtime_value_type) {
        py_object rt_offset((PyObject*)event_time::round_time_rt(
                                (PyObject*)runtime_value_type, (RuntimeValue*)offset,
                                (RuntimeValue*)rt_time_scale));
        return event_time::_new_time_rt(self, (PyObject*)event_time_type, prev,
                                        (RuntimeValue*)rt_offset.get(), cond, wait_for);
    }
    else {
        auto coffset = event_time::round_time_int(offset);
        return event_time::_new_time_int(self, (PyObject*)event_time_type, prev,
                                         coffset, false, cond, wait_for);
    }
}

template<typename TimeStep, typename RuntimeValue, typename SubSeq, typename EventTime>
static inline __attribute__((returns_nonnull)) TimeStep*
add_time_step(SubSeq *self, PyObject *cond, EventTime *start_time, PyObject *length,
              TimeStep*, RuntimeValue*)
{
    auto seqinfo = self->__pyx_base.seqinfo;
    py_object end_time((PyObject*)new_round_time(seqinfo->time_mgr, start_time, length,
                                                 cond, (EventTime*)Py_None,
                                                 (RuntimeValue*)nullptr));
    py_object o(pytype_genericalloc(timestep_type));
    auto step = (TimeStep*)o.get();
    auto seq = &step->__pyx_base;
    new (&step->actions) std::vector<py_object>();
    seq->seqinfo = py_newref(seqinfo);
    seq->start_time = py_newref(start_time);
    seq->end_time = (EventTime*)end_time.release();
    seq->cond = py_newref(cond);
    seq->C = py_newref(self->__pyx_base.C);
    step->length = py_newref(length);
    seqinfo->bt_tracker.record(event_time_key(seq->end_time));
    pylist_append(self->sub_seqs, o);
    return (TimeStep*)o.release();
}

template<typename RuntimeValue, typename SubSeq, typename EventTime>
static inline __attribute__((returns_nonnull)) SubSeq*
add_custom_step(SubSeq *self, PyObject *cond, EventTime *start_time, PyObject *cb,
                RuntimeValue*, size_t nargs=0, PyObject *const *args=nullptr,
                PyObject *kwargs=nullptr)
{
    py_object sub_seqs(pylist_new(0));
    auto seqinfo = self->__pyx_base.seqinfo;
    py_object o(pytype_genericalloc(subseq_type));
    auto subseq = (SubSeq*)o.get();
    auto seq = &subseq->__pyx_base;
    seq->seqinfo = py_newref(seqinfo);
    seq->start_time = py_newref(start_time);
    seq->end_time = py_newref(start_time);
    seq->cond = py_newref(cond);
    seq->C = py_newref(self->__pyx_base.C);
    subseq->sub_seqs = sub_seqs.release();
    subseq->dummy_step = (decltype(subseq->dummy_step))py_newref(Py_None);
    if (nargs || kwargs) {
        py_object full_args(pytuple_new(nargs + 1));
        PyTuple_SET_ITEM(full_args.get(), 0, py_newref(o.get()));
        for (auto i = 0; i < nargs; i++)
            PyTuple_SET_ITEM(full_args.get(), i + 1, py_newref(args[i]));
        py_object res(throw_if_not(pyobject_call(cb, full_args, kwargs)));
    }
    else {
        PyObject *callargs[] = { o };
        py_object res(throw_if_not(_PyObject_Vectorcall(cb, callargs, 1, nullptr)));
    }
    pylist_append(self->sub_seqs, o);
    return (SubSeq*)o.release();
}

static void type_add_method(PyTypeObject *type, PyMethodDef *meth)
{
    py_object descr(throw_if_not(PyDescr_NewMethod(type, meth)));
    throw_if(PyDict_SetItemString(type->tp_dict, meth->ml_name, descr));
}

[[noreturn]] static void raise_too_few_args(const char* func_name, bool exact,
                                            Py_ssize_t num_min, Py_ssize_t num_found)
{
    const char *more_or_less = exact ? "exactly" : "at least";
    py_throw_format(PyExc_TypeError,
                    "%.200s() takes %.8s %zd positional argument%.1s (%zd given)",
                    func_name, more_or_less, num_min,
                    (num_min == 1) ? "" : "s", num_found);
}

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
                        kws.reset(pydict_new());
                    throw_if(PyDict_SetItem(kws, name, value));
                }
            }
        }
    }
};

template<typename RuntimeValue>
struct CondCombiner {
    PyObject *cond{nullptr};
    bool needs_free{false};
    CondCombiner(PyObject *cond1, PyObject *cond2)
    {
        auto [_cond, _needs_free] = _combine_cond(cond1, cond2,
                                                  (RuntimeValue*)nullptr);
        cond = _cond;
        needs_free = _needs_free;
    }
    PyObject *take_cond()
    {
        if (needs_free) {
            needs_free = false;
            return cond;
        }
        return py_newref(cond);
    }
    ~CondCombiner()
    {
        if (needs_free) {
            Py_DECREF(cond);
        }
    }
};

template<bool is_cond>
static inline auto condseq_get_subseq(auto *self)
{
    if constexpr (is_cond) {
        return self->seq;
    }
    else {
        return self;
    }
}

template<bool is_cond>
static inline auto condseq_get_cond(auto *self)
{
    if constexpr (is_cond) {
        return self->cond;
    }
    else {
        return self->__pyx_base.cond;
    }
}

template<bool is_cond>
static inline auto condseq_get_C(auto *self)
{
    if constexpr (is_cond) {
        return self->C;
    }
    else {
        return self->__pyx_base.C;
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

template<typename CondSeq, typename TimeSeq, typename TimeStep, typename RuntimeValue,
         bool is_cond, AddStepType type>
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
        auto time_mgr = subseq->__pyx_base.seqinfo->time_mgr;
        auto new_time = event_time::_new_time_int(time_mgr, (PyObject*)event_time_type,
                                                  (EventTime*)Py_None, 0, true, cond,
                                                  (EventTime*)Py_None);
        start_time.reset((PyObject*)new_time);
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
        auto res = pytuple_new(tuple_nargs);
        auto *tuple_args = args + nargs_min;
        for (auto i = 0; i < tuple_nargs; i++)
            PyTuple_SET_ITEM(res, i, py_newref(tuple_args[i]));
        return res;
    };

    py_object kws;
    if (kwnames) {
        kws.reset(pydict_new());
        auto kwvalues = args + nargs;
        int nkws = (int)PyTuple_GET_SIZE(kwnames);
        for (int i = 0; i < nkws; i++) {
            throw_if(PyDict_SetItem(kws, PyTuple_GET_ITEM(kwnames, i), kwvalues[i]));
        }
    }

    PyObject *res;
    if (Py_TYPE(first_arg)->tp_call) {
        res = (PyObject*)add_custom_step(subseq, cond, (EventTime*)start_time.get(),
                                         first_arg, (RuntimeValue*)nullptr,
                                         tuple_nargs, args + nargs_min, kws.get());
    }
    else if (kws) {
        py_object arg_tuple(get_args_tuple());
        return PyErr_Format(PyExc_ValueError,
                            "Unexpected arguments when creating new time step, %S, %S.",
                            arg_tuple.get(), kws.get());
    }
    else if (tuple_nargs == 0) {
        res = (PyObject*)add_time_step(subseq, cond, (EventTime*)start_time.get(),
                                       first_arg, (TimeStep*)nullptr,
                                       (RuntimeValue*)nullptr);
    }
    else {
        py_object arg_tuple(get_args_tuple());
        return PyErr_Format(PyExc_ValueError,
                            "Unexpected arguments when creating new time step, %S.",
                            arg_tuple.get());
    }
    if (type == AddStepType::Step)
        pyassign(subseq->__pyx_base.end_time, ((TimeSeq*)res)->end_time);
    return res;
}
catch (...) {
    return nullptr;
}

template<typename CondSeq, typename RuntimeValue,
         bool is_cond, bool is_step=false, bool is_pulse=false>
static PyObject *condseq_set(PyObject *py_self, PyObject *const *args,
                             Py_ssize_t nargs, PyObject *kwnames) try
{
    seq_set_params params(args, nargs, kwnames, is_pulse);
    auto self = (CondSeq*)py_self;
    auto subseq = condseq_get_subseq<is_cond>(self);
    auto cond = condseq_get_cond<is_cond>(self);
    CondCombiner<RuntimeValue> cc(cond, params.cond);
    if constexpr (is_step)
        throw_if(__pyx_f_14brassboard_seq_3seq_timestep_set(
                     subseq, params.chn, params.value, cc.cond, is_pulse,
                     params.exact_time, params.kwargs()));
    else
        throw_if(__pyx_f_14brassboard_seq_3seq_subseq_set(
                     subseq, params.chn, params.value,
                     cc.cond, params.exact_time, params.kwargs()));
    return py_newref(py_self);
}
catch (...) {
    return nullptr;
}

template<typename CondSeq, typename ConditionalWrapper,
         typename RuntimeValue, bool is_cond>
static PyObject *condseq_conditional(PyObject *py_self, PyObject *const *args,
                                     Py_ssize_t nargs) try
{
    if (nargs != 1)
        raise_too_few_args("conditional", true, 1, nargs);
    auto self = (CondSeq*)py_self;
    auto subseq = condseq_get_subseq<is_cond>(self);
    auto cond = condseq_get_cond<is_cond>(self);
    CondCombiner<RuntimeValue> cc(cond, args[0]);
    auto o = pytype_genericalloc(condwrapper_type);
    auto wrapper = (ConditionalWrapper*)o;
    wrapper->seq = py_newref(subseq);
    wrapper->cond = cc.take_cond();
    wrapper->C = py_newref(condseq_get_C<is_cond>(self));
    return o;
}
catch (...) {
    return nullptr;
}

template<typename TimeStep, typename RuntimeValue>
static inline void
update_timestep(TimeStep*, RuntimeValue*)
{
    static PyMethodDef timestep_set_method = {
        "set", (PyCFunction)(void*)condseq_set<TimeStep,RuntimeValue,false,true,false>,
        METH_FASTCALL|METH_KEYWORDS, 0};
    static PyMethodDef timestep_pulse_method = {
        "pulse", (PyCFunction)(void*)condseq_set<TimeStep,RuntimeValue,false,true,true>,
        METH_FASTCALL|METH_KEYWORDS, 0};
    type_add_method(timestep_type, &timestep_set_method);
    type_add_method(timestep_type, &timestep_pulse_method);
    PyType_Modified(timestep_type);
}

template<typename SubSeq, typename ConditionalWrapper, typename TimeSeq,
         typename TimeStep, typename RuntimeValue>
static inline void update_subseq(SubSeq*, ConditionalWrapper*, TimeSeq*, TimeStep*,
                                 RuntimeValue*)
{
    static PyMethodDef subseq_conditional_method = {
        "conditional", (PyCFunction)(void*)condseq_conditional<SubSeq,ConditionalWrapper,RuntimeValue,false>,
        METH_FASTCALL, 0};
    static PyMethodDef subseq_set_method = {
        "set", (PyCFunction)(void*)condseq_set<SubSeq,RuntimeValue,false>,
        METH_FASTCALL|METH_KEYWORDS, 0};
    static PyMethodDef subseq_add_step_method = {
        "add_step",
        (PyCFunction)(void*)add_step_real<SubSeq,TimeSeq,TimeStep,RuntimeValue,false,AddStepType::Step>,
        METH_FASTCALL|METH_KEYWORDS, 0};
    static PyMethodDef subseq_add_background_method = {
        "add_background",
        (PyCFunction)(void*)add_step_real<SubSeq,TimeSeq,TimeStep,RuntimeValue,false,AddStepType::Background>,
        METH_FASTCALL|METH_KEYWORDS, 0};
    static PyMethodDef subseq_add_floating_method = {
        "add_floating",
        (PyCFunction)(void*)add_step_real<SubSeq,TimeSeq,TimeStep,RuntimeValue,false,AddStepType::Floating>,
        METH_FASTCALL|METH_KEYWORDS, 0};
    static PyMethodDef subseq_add_at_method = {
        "add_at",
        (PyCFunction)(void*)add_step_real<SubSeq,TimeSeq,TimeStep,RuntimeValue,false,AddStepType::At>,
        METH_FASTCALL|METH_KEYWORDS, 0};
    type_add_method(subseq_type, &subseq_conditional_method);
    type_add_method(subseq_type, &subseq_set_method);
    type_add_method(subseq_type, &subseq_add_step_method);
    type_add_method(subseq_type, &subseq_add_background_method);
    type_add_method(subseq_type, &subseq_add_floating_method);
    type_add_method(subseq_type, &subseq_add_at_method);
    PyType_Modified(subseq_type);
}

template<typename ConditionalWrapper, typename TimeSeq, typename TimeStep,
         typename RuntimeValue>
static inline void
update_conditional(ConditionalWrapper*, TimeSeq*, TimeStep*, RuntimeValue*)
{
    static PyMethodDef conditional_conditional_method = {
        "conditional", (PyCFunction)(void*)condseq_conditional<ConditionalWrapper,ConditionalWrapper,RuntimeValue,true>,
        METH_FASTCALL, 0};
    static PyMethodDef conditional_set_method = {
        "set", (PyCFunction)(void*)condseq_set<ConditionalWrapper,RuntimeValue,true>,
        METH_FASTCALL|METH_KEYWORDS, 0};
    static PyMethodDef conditional_add_step_method = {
        "add_step",
        (PyCFunction)(void*)add_step_real<ConditionalWrapper,TimeSeq,TimeStep,RuntimeValue,true,AddStepType::Step>,
        METH_FASTCALL|METH_KEYWORDS, 0};
    static PyMethodDef conditional_add_background_method = {
        "add_background",
        (PyCFunction)(void*)add_step_real<ConditionalWrapper,TimeSeq,TimeStep,RuntimeValue,true,AddStepType::Background>,
        METH_FASTCALL|METH_KEYWORDS, 0};
    static PyMethodDef conditional_add_floating_method = {
        "add_floating",
        (PyCFunction)(void*)add_step_real<ConditionalWrapper,TimeSeq,TimeStep,RuntimeValue,true,AddStepType::Floating>,
        METH_FASTCALL|METH_KEYWORDS, 0};
    static PyMethodDef conditional_add_at_method = {
        "add_at",
        (PyCFunction)(void*)add_step_real<ConditionalWrapper,TimeSeq,TimeStep,RuntimeValue,true,AddStepType::At>,
        METH_FASTCALL|METH_KEYWORDS, 0};
    type_add_method(condwrapper_type, &conditional_conditional_method);
    type_add_method(condwrapper_type, &conditional_set_method);
    type_add_method(condwrapper_type, &conditional_add_step_method);
    type_add_method(condwrapper_type, &conditional_add_background_method);
    type_add_method(condwrapper_type, &conditional_add_floating_method);
    type_add_method(condwrapper_type, &conditional_add_at_method);
    PyType_Modified(condwrapper_type);
}

}
