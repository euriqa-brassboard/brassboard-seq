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

#include <algorithm>
#include <vector>

namespace brassboard_seq::seq {

static PyTypeObject *event_time_type;
static PyTypeObject *timestep_type;
static PyTypeObject *subseq_type;
static PyTypeObject *condwrapper_type;
static PyTypeObject *rampfunction_type;
static PyObject *rt_time_scale;

template<typename RuntimeValue>
static inline std::pair<PyObject*,bool>
_combine_cond(PyObject *cond1, PyObject *new_cond, RuntimeValue*)
{
    if (cond1 == Py_False)
        return { Py_False, false };
    if (!rtval::is_rtval(new_cond)) {
        if (get_value_bool(new_cond, (uintptr_t)-1)) {
            return { cond1, false };
        }
        else {
            return { Py_False, false };
        }
    }
    py_object cond2((PyObject*)rtval::rt_convert_bool((RuntimeValue*)new_cond));
    if (cond1 == Py_True)
        return { cond2.release(), true };
    assert(rtval::is_rtval(cond1));
    auto o = pytype_genericalloc(rtval::RTVal_Type);
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
    if (rtval::is_rtval(offset)) {
        py_object rt_offset((PyObject*)event_time::round_time_rt(
                                (RuntimeValue*)offset, (RuntimeValue*)rt_time_scale));
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
add_time_step(SubSeq *self, PyObject *cond, EventTime *start_time, PyObject *length)
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
        res = (PyObject*)add_time_step<TimeStep,RuntimeValue>(
            subseq, cond, (EventTime*)start_time.get(), first_arg);
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

static PyObject *py_slash = pyunicode_from_string("/");

static py_object get_channel_name(auto *seqinfo, int cid)
{
    auto path = PyList_GET_ITEM(seqinfo->channel_paths, cid);
    return py_object(throw_if_not(PyUnicode_Join(py_slash, path)));
}

static inline void
timestep_set(auto *self, PyObject *chn, PyObject *value, PyObject *cond,
             bool is_pulse, bool exact_time, py_object &&kws)
{
    auto seqinfo = self->__pyx_base.seqinfo;
    int cid;
    if (Py_TYPE(chn) == &PyLong_Type) {
        auto lcid = PyLong_AsLong(chn);
        if (lcid < 0 || lcid > PyList_GET_SIZE(seqinfo->channel_paths))
            py_throw_format(PyExc_ValueError, "Channel id %ld out of bound", lcid);
        cid = lcid;
    }
    else {
        cid = __pyx_f_14brassboard_seq_3seq__get_channel_id(seqinfo, chn);
        throw_if_not(cid >= 0);
    }
    if (cid >= self->actions.size()) {
        self->actions.resize(cid + 1);
    }
    else if (self->actions[cid]) {
        auto name = get_channel_name(seqinfo, cid);
        py_throw_format(PyExc_ValueError,
                        "Multiple actions added for the same channel "
                        "at the same time on %U.", name.get());
    }
    auto aid = seqinfo->action_counter;
    auto action = seqinfo->action_alloc.alloc(value, cond, is_pulse, exact_time,
                                              std::move(kws), aid);
    action->length = self->length;
    seqinfo->bt_tracker.record(action_key(aid));
    seqinfo->action_counter = aid + 1;
    self->actions[cid] = action;
}

template<typename RuntimeValue>
static inline void
subseq_set(auto *self, PyObject *chn, PyObject *value, PyObject *cond,
           bool exact_time, py_object &&kws)
{
    auto *step = self->dummy_step;
    using TimeStep = std::remove_reference_t<decltype(*step)>;
    auto *start_time = self->__pyx_base.end_time;
    if ((PyObject*)step == Py_None || step->__pyx_base.end_time != start_time) {
        step = add_time_step<TimeStep,RuntimeValue>(self, self->__pyx_base.cond,
                                                    start_time, pylong_cached(0));
        Py_DECREF(self->dummy_step);
        self->dummy_step = step;
        // Update the current time so that a normal step added later
        // this is treated as ordered after this set event
        // rather than at the same time.
        pyassign(self->__pyx_base.end_time, step->__pyx_base.end_time);
    }
    timestep_set(step, chn, value, cond, false, exact_time, std::move(kws));
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
        timestep_set(subseq, params.chn, params.value, cc.cond, is_pulse,
                     params.exact_time, std::move(params.kws));
    else
        subseq_set<RuntimeValue>(subseq, params.chn, params.value,
                                 cc.cond, params.exact_time, std::move(params.kws));
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

template<typename TimeStep, typename SubSeq>
static void collect_actions(SubSeq *self, std::vector<action::Action*> *actions)
{
    auto sub_seqs = self->sub_seqs;
    int n = PyList_GET_SIZE(sub_seqs);
    for (int i = 0; i < n; i++) {
        auto subseq = PyList_GET_ITEM(sub_seqs, i);
        if (Py_TYPE(subseq) != timestep_type) {
            collect_actions<TimeStep>((SubSeq*)subseq, actions);
            continue;
        }
        auto step = (TimeStep*)subseq;
        auto tid = step->__pyx_base.start_time->data.id;
        auto end_tid = step->__pyx_base.end_time->data.id;
        int nactions = step->actions.size();
        for (int chn = 0; chn < nactions; chn++) {
            auto action = step->actions[chn];
            if (!action)
                continue;
            action->tid = tid;
            action->end_tid = end_tid;
            actions[chn].push_back(action);
        }
    }
}

template<typename TimeStep, typename RampFunction,
         typename RuntimeValue, typename Seq>
static inline void seq_finalize(Seq *self, TimeStep*, RampFunction*, RuntimeValue*)
{
    using EventTime = std::remove_reference_t<decltype(*self->__pyx_base.__pyx_base.start_time)>;
    auto seqinfo = self->__pyx_base.__pyx_base.seqinfo;
    auto bt_guard = set_global_tracker(&seqinfo->bt_tracker);
    auto time_mgr = seqinfo->time_mgr;
    time_mgr->__pyx_vtab->finalize(time_mgr);
    pyassign(seqinfo->channel_name_map, Py_None); // Free up memory
    auto nchn = (int)PyList_GET_SIZE(seqinfo->channel_paths);
    auto all_actions = new std::vector<action::Action*>[nchn];
    self->all_actions.reset(all_actions);
    collect_actions<TimeStep>(&self->__pyx_base, all_actions);
    auto get_time = [event_times=time_mgr->event_times] (int tid) {
        return (EventTime*)PyList_GET_ITEM(event_times, tid);
    };
    for (int cid = 0; cid < nchn; cid++) {
        auto &actions = all_actions[cid];
        std::ranges::sort(actions, [] (auto *a1, auto *a2) {
            return a1->tid < a2->tid;
        });
        py_object value(pylong_from_long(0));
        EventTime *last_time = nullptr;
        bool last_is_start = false;
        int tid = -1;
        for (auto action: actions) {
            if (action->tid == tid) {
                // It is difficult to decide the ordering of actions
                // if multiple were added to exactly the same time points.
                // We disallow this in the same timestep and we'll also disallow
                // this here.
                auto name = get_channel_name(seqinfo, cid);
                bb_throw_format(PyExc_ValueError, action_key(action->aid),
                                "Multiple actions added for the same channel "
                                "at the same time on %U.", name.get());
            }
            tid = action->tid;
            auto start_time = get_time(tid);
            if (last_time) {
                auto o = event_time::is_ordered(last_time, start_time);
                if (o != event_time::OrderBefore &&
                    (o != event_time::OrderEqual || last_is_start)) {
                    auto name = get_channel_name(seqinfo, cid);
                    bb_throw_format(PyExc_ValueError, action_key(action->aid),
                                    "Actions on %U is not statically ordered",
                                    name.get());
                }
            }
            auto action_value = action->value.get();
            auto isramp = py_issubtype_nontrivial(Py_TYPE(action_value),
                                                  rampfunction_type);
            auto cond = action->cond.get();
            last_is_start = false;
            if (!action->is_pulse) {
                last_is_start = !isramp;
                if (cond != Py_False) {
                    py_object new_value;
                    if (isramp) {
                        auto rampf = (RampFunction*)action_value;
                        auto length = action->length;
                        auto vt = rampf->__pyx_vtab;
                        throw_if(vt->set_compile_params(rampf, length, value),
                                 action_key(action->aid));
                        new_value.reset(throw_if_not(vt->eval(rampf, length, length,
                                                              value),
                                                     action_key(action->aid)));
                    }
                    else {
                        new_value.reset(py_newref(action_value));
                    }
                    if (cond == Py_True) {
                        std::swap(value, new_value);
                    }
                    else if (new_value.get() != value.get()) {
                        assert(rtval::is_rtval(cond));
                        auto endval = rtval::_new_select((rtval::_RuntimeValue*)cond,
                                                         new_value, value);
                        value.reset((PyObject*)endval);
                    }
                }
            }
            else if (cond != Py_False && isramp) {
                auto rampf = (RampFunction*)action_value;
                throw_if(rampf->__pyx_vtab->set_compile_params(rampf, action->length,
                                                               value),
                         action_key(action->aid));
            }
            last_time = last_is_start ? start_time : get_time(action->end_tid);
            action->end_val.reset(py_newref(value.get()));
        }
    }
}

template<typename RampFunction, typename RuntimeValue, typename Seq>
static inline void seq_runtime_finalize(Seq *self, unsigned age, py_object &pyage,
                                        RampFunction*, RuntimeValue*)
{
    auto seqinfo = self->__pyx_base.__pyx_base.seqinfo;
    auto bt_guard = set_global_tracker(&seqinfo->bt_tracker);
    auto time_mgr = seqinfo->time_mgr;
    self->total_time = time_mgr->__pyx_vtab->compute_all_times(time_mgr, age, pyage);
    auto assertions = seqinfo->assertions;
    int nassert = PyList_GET_SIZE(assertions);
    for (int assert_id = 0; assert_id < nassert; assert_id++) {
        auto a = PyList_GET_ITEM(assertions, assert_id);
        auto c = (RuntimeValue*)PyTuple_GET_ITEM(a, 0);
        rtval::rt_eval_throw(c, age, pyage, assert_key(assert_id));
        if (rtval::rtval_cache(c).is_zero()) {
            bb_throw_format(PyExc_AssertionError, assert_key(assert_id),
                            "%U", PyTuple_GET_ITEM(a, 1));
        }
    }
    auto get_condval = [&] (auto *action) {
        auto cond = action->cond.get();
        if (cond == Py_True)
            return true;
        if (cond == Py_False)
            return false;
        assert(rtval::is_rtval(cond));
        try {
            rtval::rt_eval_throw((RuntimeValue*)cond, age, pyage);
            return !rtval::rtval_cache((RuntimeValue*)cond).is_zero();
        }
        catch (...) {
            bb_rethrow(action_key(action->aid));
        }
    };
    auto nchn = (int)PyList_GET_SIZE(seqinfo->channel_paths);
    for (int cid = 0; cid < nchn; cid++) {
        auto &actions = self->all_actions[cid];
        long long prev_time = 0;
        for (auto action: actions) {
            bool cond_val = get_condval(action);
            action->cond_val = cond_val;
            if (!cond_val)
                continue;
            auto action_value = action->value.get();
            auto isramp = py_issubtype_nontrivial(Py_TYPE(action_value),
                                                  rampfunction_type);
            if (isramp) {
                auto rampf = (RampFunction*)action_value;
                throw_if(rampf->__pyx_vtab->set_runtime_params(rampf, age, pyage),
                         action_key(action->aid));
            }
            else if (rtval::is_rtval(action_value)) {
                rtval::rt_eval_throw((RuntimeValue*)action_value, age, pyage,
                                     action_key(action->aid));
            }
            auto action_end_val = action->end_val.get();
            if (action_end_val != action_value && rtval::is_rtval(action_end_val)) {
                rtval::rt_eval_throw((RuntimeValue*)action_end_val, age, pyage,
                                     action_key(action->aid));
            }
            // No need to evaluate action.length since the `compute_all_times`
            // above should've done it already.
            auto start_time = time_mgr->time_values[action->tid];
            auto end_time = time_mgr->time_values[action->end_tid];
            if (prev_time > start_time || start_time > end_time)
                bb_throw_format(PyExc_ValueError, action_key(action->aid),
                                "Action time order violation");
            prev_time = (isramp || action->is_pulse) ? end_time : start_time;
        }
    }
}

}
