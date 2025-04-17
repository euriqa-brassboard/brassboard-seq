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

#include "seq.h"

#include <algorithm>

#include "structmember.h"

namespace brassboard_seq::seq {

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

static inline std::pair<PyObject*,bool>
_combine_cond(py::ptr<> cond1, py::ptr<> new_cond)
{
    if (cond1 == Py_False)
        return { Py_False, false };
    if (!is_rtval(new_cond)) {
        if (get_value_bool(new_cond, (uintptr_t)-1)) {
            return { cond1, false };
        }
        else {
            return { Py_False, false };
        }
    }
    auto cond2 = rt_convert_bool(new_cond);
    if (cond1 == Py_True)
        return { cond2.rel(), true };
    assert(is_rtval(cond1));
    auto self = py::generic_alloc<RuntimeValue>();
    self->datatype = DataType::Bool;
    // self->cache_err = EvalError::NoError;
    // self->cache_val = { .i64_val = 0 };
    self->type_ = And;
    self->age = (unsigned)-1;
    self->arg0 = (RuntimeValue*)py::newref(cond1);
    self->arg1 = cond2.rel();
    self->cb_arg2 = py::immref(Py_None);
    return { (PyObject*)self.rel(), true };
}

static inline __attribute__((returns_nonnull)) PyObject*
combine_cond(py::ptr<> cond1, py::ptr<> new_cond)
{
    auto [res, needs_free] = _combine_cond(cond1, new_cond);
    if (!needs_free)
        return py::newref(res);
    return res;
}

struct CondCombiner {
    PyObject *cond{nullptr};
    bool needs_free{false};
    CondCombiner(PyObject *cond1, PyObject *cond2)
    {
        auto [_cond, _needs_free] = _combine_cond(cond1, cond2);
        cond = _cond;
        needs_free = _needs_free;
    }
    PyObject *take_cond()
    {
        if (needs_free) {
            needs_free = false;
            return cond;
        }
        return py::newref(cond);
    }
    ~CondCombiner()
    {
        if (needs_free) {
            Py_DECREF(cond);
        }
    }
};

template<typename T>
static PyObject *generic_str(PyObject *py_self)
{
    return cxx_catch([&] {
        py::stringio io;
        ((T*)py_self)->show(io, 0);
        return io.getvalue();
    });
}

template<typename CondSeq, bool is_pulse=false>
static PyObject *condseq_set(PyObject *py_self, PyObject *const *args,
                             Py_ssize_t nargs, PyObject *kwnames) try
{
    py::check_num_arg((is_pulse ? "pulse" : "set"), nargs, 2, 2);
    auto chn = args[0];
    auto value = args[1];
    bool exact_time{false};
    PyObject *arg_cond{Py_True};
    py::dict_ref kws;
    if (kwnames) {
        auto kwvalues = args + nargs;
        for (auto [i, name]: py::tuple_iter(kwnames)) {
            auto value = kwvalues[i];
            if (PyUnicode_CompareWithASCIIString(name, "cond") == 0) {
                arg_cond = value;
            }
            else if (PyUnicode_CompareWithASCIIString(name, "exact_time") == 0) {
                exact_time = get_value_bool(value, (uintptr_t)-1);
            }
            else {
                if (!kws)
                    kws = py::new_dict();
                kws.set(name, value);
            }
        }
    }
    auto self = (CondSeq*)py_self;
    CondCombiner cc(self->cond, arg_cond);
    self->get_seq()->template set<is_pulse>(chn, value, cc.cond,
                                            exact_time, std::move(kws));
    return py::newref(py_self);
}
catch (...) {
    handle_cxx_exception();
    return nullptr;
}

template<typename CondSeq>
static PyObject *condseq_wait(PyObject *py_self, PyObject *const *args,
                              Py_ssize_t nargs, PyObject *kwnames) try
{
    py::check_num_arg("wait", nargs, 1, 1);
    py::ptr<> length = args[0];
    py::ptr<> cond{Py_True};
    if (kwnames) {
        auto kwvalues = args + nargs;
        for (auto [i, name]: py::tuple_iter(kwnames)) {
            auto value = kwvalues[i];
            if (PyUnicode_CompareWithASCIIString(name, "cond") == 0) {
                cond = value;
            }
            else {
                unexpected_kwarg_error("wait", name);
            }
        }
    }
    auto self = (CondSeq*)py_self;
    CondCombiner cc(self->cond, cond);
    self->get_seq()->wait_cond(length, cc.cond);
    Py_RETURN_NONE;
}
catch (...) {
    handle_cxx_exception();
    return nullptr;
}

template<typename CondSeq>
static PyObject *condseq_wait_for(PyObject *py_self, PyObject *const *args,
                                  Py_ssize_t nargs, PyObject *kwnames)
{
    return cxx_catch([&] {
        py::check_num_arg("wait_for", nargs, 1, 2);
        auto [offset] =
            py::parse_pos_or_kw_args<"offset">("wait_for", args + 1, nargs - 1, kwnames);
        if (!offset)
            offset = py::int_cached(0);
        auto self = (CondSeq*)py_self;
        self->get_seq()->wait_for_cond(args[0], offset, self->cond);
        Py_RETURN_NONE;
    });
}

static PyObject *condwrapper_vectorcall(ConditionalWrapper *self, PyObject *const *args,
                                        size_t _nargs, PyObject *kwnames) try {
    auto nargs = PyVectorcall_NARGS(_nargs);
    py::check_no_kwnames("__call__", kwnames);
    py::check_num_arg("__call__", nargs, 1, 1);
    // Reuse the args buffer
    auto step = self->seq->add_custom_step(self->cond, self->seq->end_time,
                                           args[0], 0, &args[1], py::tuple());
    py::assign(self->seq->end_time, step->end_time);
    return (PyObject*)step.rel();
}
catch (...) {
    handle_cxx_exception();
    return nullptr;
}

template<typename CondSeq>
static PyObject *condseq_conditional(PyObject *py_self, PyObject *const *args,
                                     Py_ssize_t nargs) try
{
    py::check_num_arg("conditional", nargs, 1, 1);
    auto self = (CondSeq*)py_self;
    auto subseq = self->get_seq();
    CondCombiner cc(self->cond, args[0]);
    auto wrapper = py::generic_alloc<ConditionalWrapper>();
    wrapper->seq = py::newref(subseq);
    wrapper->cond = cc.take_cond();
    wrapper->fptr = (void*)condwrapper_vectorcall;
    return (PyObject*)wrapper.rel();
}
catch (...) {
    handle_cxx_exception();
    return nullptr;
}

template<typename CondSeq, AddStepType type>
static PyObject *add_step_real(PyObject *py_self, PyObject *const *args,
                               Py_ssize_t nargs, PyObject *_kwnames) try
{
    auto self = (CondSeq*)py_self;
    auto subseq = self->get_seq();
    auto cond = self->cond;
    auto nargs_min = type == AddStepType::At ? 2 : 1;
    py::check_num_arg(add_step_name(type), nargs, nargs_min);

    auto first_arg = args[nargs_min - 1];
    py::ref<EventTime> start_time;
    if (type == AddStepType::Background) {
        start_time.assign(subseq->end_time);
    }
    else if (type == AddStepType::Floating) {
        auto time_mgr = subseq->seqinfo->time_mgr;
        start_time.take(time_mgr->new_int(Py_None, 0, true, cond, Py_None));
    }
    else if (type == AddStepType::At) {
        if (args[0] != Py_None && !py::typeis<event_time::EventTime>(args[0]))
            return PyErr_Format(PyExc_TypeError,
                                "Argument 'tp' has incorrect type (expected EventTime, "
                                "got %.200s)", Py_TYPE(args[0])->tp_name);
        start_time.assign(args[0]);
    }
    else {
        assert(type == AddStepType::Step);
        start_time.assign(subseq->end_time);
    }

    auto tuple_nargs = nargs - nargs_min;
    auto get_args_tuple = [&] {
        if (tuple_nargs == 0)
            return py::empty_tuple.immref();
        auto res = py::new_tuple(tuple_nargs);
        auto *tuple_args = args + nargs_min;
        for (auto i = 0; i < tuple_nargs; i++)
            res.SET(i, py::ptr(tuple_args[i]));
        return res;
    };

    py::ref<TimeSeq> res;
    if (Py_TYPE(first_arg)->tp_call) {
        assert(nargs_min >= 1);
        res.take(subseq->add_custom_step(cond, start_time, first_arg,
                                         tuple_nargs, args + nargs_min, _kwnames));
    }
    else if (auto kwnames = py::tuple(_kwnames); kwnames && kwnames.size()) {
        auto kws = py::new_dict();
        auto kwvalues = args + nargs;
        for (auto [i, name]: py::tuple_iter(kwnames))
            kws.set(name, kwvalues[i]);
        return PyErr_Format(PyExc_ValueError,
                            "Unexpected arguments when creating new time step, %S, %S.",
                            get_args_tuple(), kws);
    }
    else if (tuple_nargs == 0) {
        res.take(subseq->add_time_step(cond, start_time, first_arg));
    }
    else {
        return PyErr_Format(PyExc_ValueError,
                            "Unexpected arguments when creating new time step, %S.",
                            get_args_tuple());
    }
    if (type == AddStepType::Step)
        py::assign(subseq->end_time, res->end_time);
    return (PyObject*)res.rel();
}
catch (...) {
    handle_cxx_exception();
    return nullptr;
}

inline int SeqInfo::get_channel_id(py::str name)
{
    if (auto chn = py::dict(channel_name_map).try_get(name)) [[likely]]
        return PyLong_AsLong(chn);
    auto path = py::tuple_ref(config->translate_channel(name));
    if (auto chn = py::dict(channel_path_map).try_get(path)) {
        py::dict(channel_name_map).set(name, chn);
        return PyLong_AsLong(chn);
    }
    int cid = py::list(channel_paths).size();
    py::list(channel_paths).append(path);
    auto pycid = py::new_int(cid);
    py::dict(channel_path_map).set(path, pycid);
    py::dict(channel_name_map).set(name, pycid);
    return cid;
}

__attribute__((visibility("protected")))
PyTypeObject SeqInfo::Type = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "brassboard_seq.seq.SeqInfo",
    .tp_basicsize = sizeof(SeqInfo),
    .tp_dealloc = [] (PyObject *py_self) {
        auto self = (SeqInfo*)py_self;
        PyObject_GC_UnTrack(py_self);
        Type.tp_clear(py_self);
        call_destructor(&self->bt_tracker);
        call_destructor(&self->action_alloc);
        Py_TYPE(py_self)->tp_free(py_self);
    },
    .tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_HAVE_GC,
    .tp_traverse = [] (PyObject *py_self, visitproc visit, void *arg) {
        auto self = (SeqInfo*)py_self;
        Py_VISIT(self->time_mgr);
        Py_VISIT(self->assertions);
        Py_VISIT(self->config);
        Py_VISIT(self->C);
        return 0;
    },
    .tp_clear = [] (PyObject *py_self) {
        auto self = (SeqInfo*)py_self;
        Py_CLEAR(self->time_mgr);
        Py_CLEAR(self->assertions);
        Py_CLEAR(self->config);
        Py_CLEAR(self->channel_name_map);
        Py_CLEAR(self->channel_path_map);
        Py_CLEAR(self->channel_paths);
        Py_CLEAR(self->C);
        return 0;
    },
};

inline void TimeSeq::show_cond_suffix(py::stringio &io) const
{
    if (cond != Py_True) {
        io.write_ascii(" if ");
        io.write_str(cond);
    }
    io.write_ascii("\n");
}

inline void TimeSeq::dealloc()
{
    PyObject_GC_UnTrack(this);
    Py_TYPE(this)->tp_free(this);
}

inline int TimeSeq::traverse(visitproc visit, void *arg)
{
    Py_VISIT(seqinfo);
    Py_VISIT(start_time);
    Py_VISIT(end_time);
    Py_VISIT(cond);
    return 0;
}

inline void TimeSeq::clear()
{
    Py_CLEAR(seqinfo);
    Py_CLEAR(start_time);
    Py_CLEAR(end_time);
    Py_CLEAR(cond);
}

inline void TimeSeq::cclear()
{
}

static PyObject *get_channel_id(PyObject *self, PyObject *name)
{
    return cxx_catch([&] {
        auto id = ((TimeSeq*)self)->seqinfo->get_channel_id(name);
        return py::new_int(id);
    });
}

static PyObject *set_time(PyObject *py_self, PyObject *const *args,
                          Py_ssize_t nargs, PyObject *kwnames)
{
    return cxx_catch([&] {
        py::check_num_arg("set_time", nargs, 1, 2);
        auto [offset] =
            py::parse_pos_or_kw_args<"offset">("set_time", args + 1, nargs - 1, kwnames);
        auto time = (args[0] == Py_None ? py::ptr<EventTime>(Py_None) :
                     py::arg_cast<EventTime>(args[0], "time"));
        if (!offset)
            offset = py::int_cached(0);
        auto self = (TimeSeq*)py_self;
        if (is_rtval(offset))
            self->start_time->set_base_rt(time, event_time::round_time_rt(offset));
        else
            self->start_time->set_base_int(time, event_time::round_time_int(offset));
        self->seqinfo->bt_tracker.record(event_time_key(self->start_time));
        Py_RETURN_NONE;
    });
}

static PyObject *rt_assert(PyObject *py_self, PyObject *const *args,
                           Py_ssize_t nargs, PyObject *kwnames)
{
    return cxx_catch([&] {
        py::check_num_arg("rt_assert", nargs, 1, 2);
        auto [msg] = py::parse_pos_or_kw_args<"msg">("rt_assert", args + 1,
                                                     nargs - 1, kwnames);
        auto c = py::ptr(args[0]);
        if (!msg)
            msg = "Assertion failed"_py;
        if (is_rtval(c)) {
            auto seqinfo = ((TimeSeq*)py_self)->seqinfo;
            auto assertions = py::list(seqinfo->assertions);
            seqinfo->bt_tracker.record(assert_key(assertions.size()));
            auto a = py::new_tuple(2);
            a.SET(0, c);
            a.SET(1, msg);
            assertions.append(std::move(a));
        }
        else if (!get_value_bool(c, -1)) {
            py_throw_format(PyExc_AssertionError, "%U", msg);
        }
        Py_RETURN_NONE;
    });
}

#define py_offsetof(type, member) [] () constexpr {     \
        type v;                                         \
        return ((char*)&v.member) - (char*)&v;          \
    } ()

static PyMemberDef TimeSeq_members[] = {
    {"start_time", T_OBJECT_EX, py_offsetof(TimeSeq, start_time), READONLY},
    {"end_time", T_OBJECT_EX, py_offsetof(TimeSeq, end_time), READONLY},
    {}
};

static PyGetSetDef TimeSeq_getsets[] = {
    {"C", [] (PyObject *py_self, void*) -> PyObject* {
        return py::newref(((TimeSeq*)py_self)->seqinfo->C); }},
    {}
};

__attribute__((visibility("protected")))
PyTypeObject TimeSeq::Type = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "brassboard_seq.seq.TimeSeq",
    .tp_basicsize = sizeof(TimeSeq),
    .tp_dealloc = [] (PyObject *py_self) {
        ((TimeSeq*)py_self)->clear();
        ((TimeSeq*)py_self)->cclear();
        ((TimeSeq*)py_self)->dealloc();
    },
    .tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE|Py_TPFLAGS_HAVE_GC,
    .tp_traverse = [] (PyObject *py_self, visitproc visit, void *arg) {
        return ((TimeSeq*)py_self)->traverse(visit, arg);
    },
    .tp_clear = [] (PyObject *py_self) {
        ((TimeSeq*)py_self)->clear();
        return 0;
    },
    .tp_methods = (PyMethodDef[]){
        {"get_channel_id", (PyCFunction)(void*)get_channel_id, METH_O, 0},
        {"set_time", (PyCFunction)(void*)set_time, METH_FASTCALL|METH_KEYWORDS, 0},
        {"rt_assert", (PyCFunction)(void*)rt_assert, METH_FASTCALL|METH_KEYWORDS, 0},
        {0, 0, 0, 0}
    },
    .tp_members = TimeSeq_members,
    .tp_getset = TimeSeq_getsets,
};

inline void TimeStep::cclear()
{
    TimeSeq::cclear();
    call_destructor(&actions);
}

inline int TimeStep::traverse(visitproc visit, void *arg)
{
    Py_VISIT(length);
    return TimeSeq::traverse(visit, arg);
}

inline void TimeStep::clear()
{
    TimeSeq::clear();
    Py_CLEAR(length);
}

template<bool is_pulse>
inline void TimeStep::set(py::ptr<> chn, py::ptr<> value, py::ptr<> cond,
                          bool exact_time, py::dict_ref &&kws)
{
    int cid;
    if (chn.typeis<py::int_>()) {
        auto lcid = PyLong_AsLong(chn);
        if (lcid < 0 || lcid > py::list(seqinfo->channel_paths).size())
            py_throw_format(PyExc_ValueError, "Channel id %ld out of bound", lcid);
        cid = lcid;
    }
    else {
        cid = seqinfo->get_channel_id(chn);
    }
    if (cid >= actions.size()) {
        actions.resize(cid + 1);
    }
    else if (actions[cid]) {
        py_throw_format(PyExc_ValueError,
                        "Multiple actions added for the same channel "
                        "at the same time on %U.", seqinfo->channel_name_from_id(cid));
    }
    auto aid = seqinfo->action_counter;
    auto action = seqinfo->action_alloc.alloc(value, cond, is_pulse, exact_time,
                                              std::move(kws), aid);
    action->length = length;
    seqinfo->bt_tracker.record(action_key(aid));
    seqinfo->action_counter = aid + 1;
    actions[cid] = action;
}

inline void TimeStep::show(py::stringio &io, int indent) const
{
    io.write_rep_ascii(indent, " ");
    io.write_ascii("TimeStep(");
    io.write_str(length);
    io.write_ascii(")@T[");
    std::array<char, 32> str_buff;
    auto ptr = to_chars(str_buff, start_time->data.id);
    io.write_ascii(str_buff.data(), ptr - str_buff.data());
    io.write_ascii("]");
    show_cond_suffix(io);
    int nactions = actions.size();
    for (int chn_idx = 0; chn_idx < nactions; chn_idx++) {
        auto action = actions[chn_idx];
        if (!action)
            continue;
        io.write_rep_ascii(indent + 2, " ");
        io.write(seqinfo->channel_name_from_id(chn_idx));
        io.write_ascii(": ");
        io.write(action->py_str());
        io.write_ascii("\n");
    }
}

__attribute__((visibility("protected")))
PyTypeObject TimeStep::Type = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "brassboard_seq.seq.TimeStep",
    .tp_basicsize = sizeof(TimeStep),
    .tp_dealloc = [] (PyObject *py_self) {
        ((TimeStep*)py_self)->clear();
        ((TimeStep*)py_self)->cclear();
        ((TimeStep*)py_self)->dealloc();
    },
    .tp_repr = generic_str<TimeStep>,
    .tp_str = generic_str<TimeStep>,
    .tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_HAVE_GC,
    .tp_traverse = [] (PyObject *py_self, visitproc visit, void *arg) {
        return ((TimeStep*)py_self)->traverse(visit, arg);
    },
    .tp_clear = [] (PyObject *py_self) {
        ((TimeStep*)py_self)->clear();
        return 0;
    },
    .tp_methods = (PyMethodDef[]){
        {"set", (PyCFunction)(void*)condseq_set<TimeStep,false>,
         METH_FASTCALL|METH_KEYWORDS, 0},
        {"pulse", (PyCFunction)(void*)condseq_set<TimeStep,true>,
         METH_FASTCALL|METH_KEYWORDS, 0},
        {0, 0, 0, 0}
    },
    .tp_base = &TimeSeq::Type,
};

inline void SubSeq::cclear()
{
    TimeSeq::cclear();
}

inline int SubSeq::traverse(visitproc visit, void *arg)
{
    Py_VISIT(sub_seqs);
    Py_VISIT(dummy_step);
    return TimeSeq::traverse(visit, arg);
}

inline void SubSeq::clear()
{
    TimeSeq::clear();
    Py_CLEAR(sub_seqs);
    Py_CLEAR(dummy_step);
}

inline void SubSeq::show_subseqs(py::stringio &io, int indent) const
{
    for (auto [i, seq]: py::list_iter(sub_seqs)) {
        if (auto step = py::exact_cast<TimeStep>(seq)) {
            step->show(io, indent);
        }
        else {
            ((SubSeq*)seq)->show(io, indent);
        }
    }
}

inline void SubSeq::show(py::stringio &io, int indent) const
{
    io.write_rep_ascii(indent, " ");
    io.write_ascii("SubSeq@T[");
    std::array<char, 32> str_buff;
    auto ptr = to_chars(str_buff, start_time->data.id);
    io.write_ascii(str_buff.data(), ptr - str_buff.data());
    io.write_ascii("] - T[");
    ptr = to_chars(str_buff, end_time->data.id);
    io.write_ascii(str_buff.data(), ptr - str_buff.data());
    io.write_ascii("]");
    show_cond_suffix(io);
    show_subseqs(io, indent + 2);
}

template<bool is_pulse>
inline void SubSeq::set(py::ptr<> chn, py::ptr<> value, py::ptr<> cond,
                        bool exact_time, py::dict_ref &&kws)
{
    static_assert(!is_pulse);
    if ((PyObject*)dummy_step == Py_None || dummy_step->end_time != end_time) {
        auto new_step = add_time_step(cond, end_time, py::int_cached(0));
        py::assign(dummy_step, std::move(new_step));
        // Update the current time so that a normal step added later
        // this is treated as ordered after this set event
        // rather than at the same time.
        py::assign(end_time, dummy_step->end_time);
    }
    dummy_step->set<false>(chn, value, cond, exact_time, std::move(kws));
}

inline void SubSeq::wait_cond(py::ptr<> length, py::ptr<> cond)
{
    auto new_time = seqinfo->time_mgr->new_round(end_time, length, cond, Py_None);
    seqinfo->bt_tracker.record(event_time_key(new_time));
    py::assign(end_time, std::move(new_time));
}

inline void SubSeq::wait_for_cond(py::ptr<> _tp0, py::ptr<> offset, py::ptr<> cond)
{
    auto tp0 = py::cast<EventTime>(_tp0);
    if (!tp0)
        tp0 = py::arg_cast<TimeSeq>(_tp0, "time_point")->end_time;
    auto new_time = seqinfo->time_mgr->new_round(end_time, offset, cond, tp0);
    seqinfo->bt_tracker.record(event_time_key(new_time));
    py::assign(end_time, std::move(new_time));
}

inline py::ref<SubSeq>
SubSeq::add_custom_step(py::ptr<> cond, py::ptr<EventTime> start_time, py::ptr<> cb,
                        size_t nargs, PyObject *const *args, py::tuple kwnames)
{
    auto subseq = py::generic_alloc<SubSeq>();
    subseq->seqinfo = py::newref(seqinfo);
    subseq->start_time = py::newref(start_time);
    subseq->end_time = py::newref(start_time);
    subseq->cond = py::newref(cond);
    subseq->sub_seqs = py::new_list(0).rel();
    subseq->dummy_step = (TimeStep*)py::immref(Py_None);
    // The python vectorcall ABI allows us to temporarily change the argument array
    // as long as we restore it before returning.
    auto prev_arg = args[-1];
    ((PyObject**)args)[-1] = (PyObject*)subseq;
    ScopeExit restore_arg([&] { ((PyObject**)args)[-1] = prev_arg; });
    cb.vcall(&args[-1], nargs + 1, kwnames);
    py::list(sub_seqs).append(subseq);
    return subseq;
}

inline py::ref<TimeStep>
SubSeq::add_time_step(py::ptr<> cond, py::ptr<EventTime> start_time, py::ptr<> length)
{
    auto end_time = seqinfo->time_mgr->new_round(start_time, length, cond, Py_None);
    auto step = py::generic_alloc<TimeStep>();
    call_constructor(&step->actions);
    step->seqinfo = py::newref(seqinfo);
    step->start_time = py::newref(start_time);
    step->end_time = end_time.rel();
    step->cond = py::newref(cond);
    step->length = py::newref(length);
    seqinfo->bt_tracker.record(event_time_key(step->end_time));
    py::list(sub_seqs).append(step);
    return step;
}

static PyMemberDef SubSeq_members[] = {
    {"current_time", T_OBJECT_EX, py_offsetof(SubSeq, end_time), READONLY},
    {}
};

__attribute__((visibility("protected")))
PyTypeObject SubSeq::Type = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "brassboard_seq.seq.SubSeq",
    .tp_basicsize = sizeof(SubSeq),
    .tp_dealloc = [] (PyObject *py_self) {
        ((SubSeq*)py_self)->clear();
        ((SubSeq*)py_self)->cclear();
        ((SubSeq*)py_self)->dealloc();
    },
    .tp_repr = generic_str<SubSeq>,
    .tp_str = generic_str<SubSeq>,
    .tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE|Py_TPFLAGS_HAVE_GC,
    .tp_traverse = [] (PyObject *py_self, visitproc visit, void *arg) {
        return ((SubSeq*)py_self)->traverse(visit, arg);
    },
    .tp_clear = [] (PyObject *py_self) {
        ((SubSeq*)py_self)->clear();
        return 0;
    },
    .tp_methods = (PyMethodDef[]){
        {"wait", (PyCFunction)(void*)condseq_wait<SubSeq>,
         METH_FASTCALL|METH_KEYWORDS, 0},
        {"wait_for", (PyCFunction)(void*)condseq_wait_for<SubSeq>,
         METH_FASTCALL|METH_KEYWORDS, 0},
        {"conditional", (PyCFunction)(void*)condseq_conditional<SubSeq>,
         METH_FASTCALL, 0},
        {"set", (PyCFunction)(void*)condseq_set<SubSeq>,
         METH_FASTCALL|METH_KEYWORDS, 0},
        {"add_step",
         (PyCFunction)(void*)add_step_real<SubSeq,AddStepType::Step>,
         METH_FASTCALL|METH_KEYWORDS, 0},
        {"add_background",
         (PyCFunction)(void*)add_step_real<SubSeq,AddStepType::Background>,
         METH_FASTCALL|METH_KEYWORDS, 0},
        {"add_floating",
         (PyCFunction)(void*)add_step_real<SubSeq,AddStepType::Floating>,
         METH_FASTCALL|METH_KEYWORDS, 0},
        {"add_at",
         (PyCFunction)(void*)add_step_real<SubSeq,AddStepType::At>,
         METH_FASTCALL|METH_KEYWORDS, 0},
        {0, 0, 0, 0}
    },
    .tp_members = SubSeq_members,
    .tp_base = &TimeSeq::Type,
};

inline void ConditionalWrapper::show(py::stringio &io, int indent) const
{
    io.write_rep_ascii(indent, " ");
    io.write_ascii("ConditionalWrapper(");
    io.write_str(cond);
    io.write_ascii(") for\n");
    if (auto s = py::exact_cast<Seq>(seq))
        return s->show(io, indent + 2);
    seq->show(io, indent + 2);
}

static PyGetSetDef ConditionalWrapper_getsets[] = {
    {"C", [] (PyObject *py_self, void*) -> PyObject* {
        return py::newref(((ConditionalWrapper*)py_self)->seq->seqinfo->C); }},
    {}
};

__attribute__((visibility("protected")))
PyTypeObject ConditionalWrapper::Type = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "brassboard_seq.seq.ConditionalWrapper",
    .tp_basicsize = sizeof(ConditionalWrapper),
    .tp_dealloc = [] (PyObject *py_self) {
        PyObject_GC_UnTrack(py_self);
        Type.tp_clear(py_self);
        Py_TYPE(py_self)->tp_free(py_self);
    },
    .tp_vectorcall_offset = py_offsetof(ConditionalWrapper, fptr),
    .tp_repr = generic_str<ConditionalWrapper>,
    .tp_call = PyVectorcall_Call,
    .tp_str = generic_str<ConditionalWrapper>,
    .tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_HAVE_GC,
    .tp_traverse = [] (PyObject *py_self, visitproc visit, void *arg) {
        auto self = (ConditionalWrapper*)py_self;
        Py_VISIT(self->seq);
        Py_VISIT(self->cond);
        return 0;
    },
    .tp_clear = [] (PyObject *py_self) {
        auto self = (ConditionalWrapper*)py_self;
        Py_CLEAR(self->seq);
        Py_CLEAR(self->cond);
        return 0;
    },
    .tp_methods = (PyMethodDef[]){
        {"wait", (PyCFunction)(void*)condseq_wait<ConditionalWrapper>,
         METH_FASTCALL|METH_KEYWORDS, 0},
        {"wait_for", (PyCFunction)(void*)condseq_wait_for<ConditionalWrapper>,
         METH_FASTCALL|METH_KEYWORDS, 0},
        {"conditional",
         (PyCFunction)(void*)condseq_conditional<ConditionalWrapper>,
         METH_FASTCALL, 0},
        {"set", (PyCFunction)(void*)condseq_set<ConditionalWrapper>,
         METH_FASTCALL|METH_KEYWORDS, 0},
        {"add_step",
         (PyCFunction)(void*)add_step_real<ConditionalWrapper,AddStepType::Step>,
         METH_FASTCALL|METH_KEYWORDS, 0},
        {"add_background",
         (PyCFunction)(void*)add_step_real<ConditionalWrapper,AddStepType::Background>,
         METH_FASTCALL|METH_KEYWORDS, 0},
        {"add_floating",
         (PyCFunction)(void*)add_step_real<ConditionalWrapper,AddStepType::Floating>,
         METH_FASTCALL|METH_KEYWORDS, 0},
        {"add_at",
         (PyCFunction)(void*)add_step_real<ConditionalWrapper,AddStepType::At>,
         METH_FASTCALL|METH_KEYWORDS, 0},
        {0, 0, 0, 0}
    },
    .tp_getset = ConditionalWrapper_getsets,
};

inline void Seq::show(py::stringio &io, int indent) const
{
    io.write_rep_ascii(indent, " ");
    io.write_ascii("Seq - T[");
    std::array<char, 32> str_buff;
    auto ptr = to_chars(str_buff, end_time->data.id);
    io.write_ascii(str_buff.data(), ptr - str_buff.data());
    io.write_ascii("]\n");
    for (auto [i, t]: py::list_iter(seqinfo->time_mgr->event_times)) {
        io.write_rep_ascii(indent + 1, " ");
        io.write_ascii("T[");
        auto ptr = to_chars(str_buff, i);
        io.write_ascii(str_buff.data(), ptr - str_buff.data());
        io.write_ascii("]: ");
        io.write_str(t);
        io.write_ascii("\n");
    }
    show_subseqs(io, indent + 2);
}

__attribute__((visibility("protected")))
PyTypeObject Seq::Type = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "brassboard_seq.seq.Seq",
    .tp_basicsize = sizeof(Seq),
    .tp_dealloc = [] (PyObject *py_self) {
        ((Seq*)py_self)->clear();
        ((Seq*)py_self)->cclear();
        ((Seq*)py_self)->dealloc();
    },
    .tp_repr = generic_str<Seq>,
    .tp_str = generic_str<Seq>,
    .tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_HAVE_GC,
    .tp_traverse = [] (PyObject *py_self, visitproc visit, void *arg) {
        return ((Seq*)py_self)->traverse(visit, arg);
    },
    .tp_clear = [] (PyObject *py_self) {
        ((Seq*)py_self)->clear();
        return 0;
    },
    .tp_base = &SubSeq::Type,
    .tp_vectorcall = [] (PyObject *type, PyObject *const *args, size_t _nargs,
                         PyObject *kwnames) -> PyObject* {
        auto nargs = PyVectorcall_NARGS(_nargs);
        return cxx_catch([&] {
            py::check_num_arg("Seq.__init__", nargs, 1, 2);
            auto [py_max_frame] =
                py::parse_pos_or_kw_args<"max_frame">("Seq.__init__", args + 1,
                                                      nargs - 1, kwnames);
            int max_frame = 0;
            if (py_max_frame) {
                max_frame = PyLong_AsLong(py_max_frame);
                if (max_frame < 0) {
                    throw_pyerr();
                    py_throw_format(PyExc_ValueError, "max_frame cannot be negative");
                }
            }
            auto self = py::generic_alloc<Seq>();
            self->start_time = (EventTime*)py::newref(Py_None);
            self->cond = py::immref(Py_True);
            self->sub_seqs = py::new_list(0).rel();
            self->dummy_step = (TimeStep*)py::immref(Py_None);
            auto seqinfo = py::generic_alloc<SeqInfo>();
            call_constructor(&seqinfo->bt_tracker);
            seqinfo->bt_tracker.max_frame = max_frame;
            call_constructor(&seqinfo->action_alloc);
            seqinfo->action_counter = 0;
            seqinfo->config = py::newref(py::arg_cast<config::Config>(args[0], "config"));
            seqinfo->time_mgr = event_time::TimeManager::alloc();
            seqinfo->assertions = py::new_list(0).rel();
            seqinfo->channel_name_map = py::new_dict().rel();
            seqinfo->channel_path_map = py::new_dict().rel();
            seqinfo->channel_paths = py::new_list(0).rel();
            seqinfo->C = scan::ParamPack::new_empty();
            self->end_time = seqinfo->time_mgr->new_int(Py_None, 0, false,
                                                        Py_True, Py_None).rel();
            seqinfo->bt_tracker.record(event_time_key(self->end_time));
            self->seqinfo = seqinfo.rel();
            return self;
        });
    },
};

__attribute__((constructor))
static void init()
{
    throw_if(PyType_Ready(&SeqInfo::Type) < 0);
    throw_if(PyType_Ready(&TimeSeq::Type) < 0);
    throw_if(PyType_Ready(&TimeStep::Type) < 0);
    throw_if(PyType_Ready(&SubSeq::Type) < 0);
    throw_if(PyType_Ready(&ConditionalWrapper::Type) < 0);
    throw_if(PyType_Ready(&Seq::Type) < 0);
}

}
