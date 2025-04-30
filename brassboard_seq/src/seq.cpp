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

namespace brassboard_seq::seq {

enum class AddStepType {
    Step,
    Background,
    Floating,
    At,
};

template<typename CondSeq, AddStepType type>
static inline auto add_step_name()
{
    if constexpr (type == AddStepType::Background) {
        return CondSeq::ClsName + ".add_background";
    }
    else if constexpr (type == AddStepType::Floating) {
        return CondSeq::ClsName + ".add_floating";
    }
    else if constexpr (type == AddStepType::At) {
        return CondSeq::ClsName + ".add_at";
    }
    else {
        static_assert(type == AddStepType::Step);
        return CondSeq::ClsName + ".add_step";
    }
}

template<typename T> static constexpr auto seq_dealloc = py::tp_cxx_dealloc<true,T>;

template<typename T>
static constexpr auto seq_clear = py::tp_field_pack_clear<typename T::fields>;
template<typename T>
static constexpr auto seq_traverse = py::tp_field_pack_traverse<typename T::fields>;

struct CondCombiner {
    PyObject *cond;
    bool needs_free{false};
    CondCombiner(py::ptr<> cond1, py::ptr<> new_cond)
    {
        if (cond1 == Py_False) {
            cond = Py_False;
            return;
        }
        assert(cond1 == Py_True || is_rtval(cond1));
        if (!is_rtval(new_cond)) {
            cond = new_cond.as_bool() ? cond1.get() : Py_False;
            return;
        }
        auto cond2 = rt_convert_bool(new_cond);
        if (cond1 != Py_True)
            cond2 = RuntimeValue::alloc(And, DataType::Bool, cond1,
                                        std::move(cond2), py::new_none());
        needs_free = true;
        cond = cond2.rel();
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
            py::DECREF(cond);
        }
    }
};

template<typename T>
static constexpr auto seq_str = py::unifunc<[] (py::ptr<T> self) {
    py::stringio io;
    self->show(io, 0);
    return io.getvalue();
}>;

template<typename CondSeq, bool is_pulse=false>
static auto condseq_set(py::ptr<CondSeq> self, PyObject *const *args,
                        Py_ssize_t nargs, PyObject *kwnames)
{
    py::check_num_arg((is_pulse ? CondSeq::ClsName + ".pulse" :
                       CondSeq::ClsName + ".set"), nargs, 2, 2);
    auto chn = args[0];
    auto value = args[1];
    bool exact_time{false};
    PyObject *arg_cond{Py_True};
    py::dict_ref kws;
    if (kwnames) {
        auto kwvalues = args + nargs;
        for (auto [i, name]: py::tuple_iter(kwnames)) {
            auto kwvalue = py::ptr(kwvalues[i]);
            if (PyUnicode_CompareWithASCIIString(name, "cond") == 0) {
                arg_cond = kwvalue;
            }
            else if (PyUnicode_CompareWithASCIIString(name, "exact_time") == 0) {
                exact_time = kwvalue.as_bool();
            }
            else {
                if (!kws)
                    kws = py::new_dict();
                kws.set(name, kwvalue);
            }
        }
    }
    CondCombiner cc(self->cond, arg_cond);
    self->get_seq()->template set<is_pulse>(chn, value, cc.cond,
                                            exact_time, std::move(kws));
    return self.ref();
}

template<typename CondSeq>
static void condseq_wait(py::ptr<CondSeq> self, PyObject *const *args,
                         Py_ssize_t nargs, PyObject *kwnames)
{
    py::check_num_arg(CondSeq::ClsName + ".wait", nargs, 1, 1);
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
    CondCombiner cc(self->cond, cond);
    self->get_seq()->wait_cond(length, cc.cond);
}

template<typename CondSeq>
static void condseq_wait_for(py::ptr<CondSeq> self, PyObject *const *args,
                             Py_ssize_t nargs, PyObject *kwnames)
{
    py::check_num_arg(CondSeq::ClsName + ".wait_for", nargs, 1, 2);
    auto [offset] =
        py::parse_pos_or_kw_args<"offset">("wait_for", args + 1, nargs - 1, kwnames);
    if (!offset)
        offset = py::int_cached(0);
    self->get_seq()->wait_for_cond(args[0], offset, self->cond);
}

static auto condwrapper_vectorcall(py::ptr<ConditionalWrapper> self, PyObject *const *args,
                                   ssize_t nargs, py::tuple kwnames)
{
    py::check_no_kwnames("ConditionalWrapper.__call__", kwnames);
    py::check_num_arg("ConditionalWrapper.__call__", nargs, 1, 1);
    // Reuse the args buffer
    auto step = self->seq->add_custom_step(self->cond, self->seq->end_time,
                                           args[0], 0, &args[1], py::tuple());
    self->seq->end_time.assign(step->end_time);
    return step;
}

template<typename CondSeq>
static auto condseq_conditional(py::ptr<CondSeq> self, py::ptr<> cond)
{
    auto subseq = self->get_seq();
    CondCombiner cc(self->cond, cond);
    auto wrapper = py::generic_alloc<ConditionalWrapper>();
    call_constructor(&wrapper->seq, py::newref(subseq));
    call_constructor(&wrapper->cond, cc.take_cond());
    *(void**)(&wrapper.get()[1]) = (void*)py::vectorfunc<condwrapper_vectorcall>;
    return wrapper;
}

template<typename CondSeq, AddStepType type>
static py::ref<TimeSeq> add_step_real(py::ptr<CondSeq> self, PyObject *const *args,
                                      Py_ssize_t nargs, py::tuple kwnames)
{
    auto subseq = self->get_seq();
    py::ptr cond = self->cond;
    auto nargs_min = type == AddStepType::At ? 2 : 1;
    py::check_num_arg(add_step_name<CondSeq,type>(), nargs, nargs_min);

    auto first_arg = py::ptr(args[nargs_min - 1]);
    py::ref<EventTime> start_time;
    if (type == AddStepType::Background) {
        start_time.assign(subseq->end_time);
    }
    else if (type == AddStepType::Floating) {
        start_time.take(subseq->seqinfo->time_mgr->new_int(Py_None, 0, true,
                                                           cond, Py_None));
    }
    else if (type == AddStepType::At) {
        if (args[0] != Py_None && !py::typeis<event_time::EventTime>(args[0]))
            py_throw_format(PyExc_TypeError,
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
            return py::new_tuple();
        auto res = py::new_tuple(tuple_nargs);
        auto *tuple_args = args + nargs_min;
        for (auto i = 0; i < tuple_nargs; i++)
            res.SET(i, py::ptr(tuple_args[i]));
        return res;
    };

    py::ref<TimeSeq> res;
    if (first_arg.type()->tp_call) {
        assert(nargs_min >= 1);
        res.take(subseq->add_custom_step(cond, start_time, first_arg,
                                         tuple_nargs, args + nargs_min, kwnames));
    }
    else if (kwnames && kwnames.size()) {
        auto kws = py::new_dict();
        auto kwvalues = args + nargs;
        for (auto [i, name]: py::tuple_iter(kwnames))
            kws.set(name, kwvalues[i]);
        py_throw_format(PyExc_ValueError,
                        "Unexpected arguments when creating new time step, %S, %S.",
                        get_args_tuple(), kws);
    }
    else if (tuple_nargs == 0) {
        res.take(subseq->add_time_step(cond, start_time, first_arg));
    }
    else {
        py_throw_format(PyExc_ValueError,
                        "Unexpected arguments when creating new time step, %S.",
                        get_args_tuple());
    }
    if (type == AddStepType::Step)
        subseq->end_time.assign(res->end_time);
    return res;
}

inline int SeqInfo::get_channel_id(py::str name)
{
    if (auto chn = channel_name_map.try_get(name)) [[likely]]
        return chn.as_int();
    auto path = py::tuple_ref(config->translate_channel(name));
    if (auto chn = channel_path_map.try_get(path)) {
        channel_name_map.set(name, chn);
        return chn.as_int();
    }
    int cid = channel_paths.size();
    channel_paths.append(path);
    auto pycid = py::new_int(cid);
    channel_path_map.set(path, pycid);
    channel_name_map.set(name, pycid);
    return cid;
}

__attribute__((visibility("protected")))
PyTypeObject SeqInfo::Type = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "brassboard_seq.seq.SeqInfo",
    .tp_basicsize = sizeof(SeqInfo),
    .tp_dealloc = py::tp_cxx_dealloc<true,SeqInfo>,
    .tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_HAVE_GC,
    .tp_traverse = py::tp_field_traverse<SeqInfo,&SeqInfo::time_mgr,&SeqInfo::assertions,
    &SeqInfo::config,&SeqInfo::C>,
    .tp_clear = py::tp_field_clear<SeqInfo,&SeqInfo::time_mgr,&SeqInfo::assertions,
    &SeqInfo::config,&SeqInfo::channel_name_map,&SeqInfo::channel_path_map,
    &SeqInfo::channel_paths,&SeqInfo::C>,
};

inline void TimeSeq::show_cond_suffix(py::stringio &io) const
{
    if (cond != Py_True) {
        io.write_ascii(" if ");
        io.write_str(cond);
    }
    io.write_ascii("\n");
}

static auto get_channel_id(py::ptr<TimeSeq> self, PyObject *name)
{
    auto id = self->seqinfo->get_channel_id(name);
    return py::new_int(id);
}

static void set_time(py::ptr<TimeSeq> self, PyObject *const *args,
                     Py_ssize_t nargs, PyObject *kwnames)
{
    py::check_num_arg("TimeSeq.set_time", nargs, 1, 2);
    auto [offset] =
        py::parse_pos_or_kw_args<"offset">("set_time", args + 1, nargs - 1, kwnames);
    auto time = (args[0] == Py_None ? py::ptr<EventTime>(Py_None) :
                 py::arg_cast<EventTime,true>(args[0], "time"));
    if (!offset)
        offset = py::int_cached(0);
    if (is_rtval(offset))
        self->start_time->set_base_rt(time, event_time::round_time_rt(offset));
    else
        self->start_time->set_base_int(time, event_time::round_time_int(offset));
    self->seqinfo->bt_tracker.record(event_time_key(self->start_time));
}

static void rt_assert(py::ptr<TimeSeq> self, PyObject *const *args,
                      Py_ssize_t nargs, PyObject *kwnames)
{
    py::check_num_arg("TimeSeq.rt_assert", nargs, 1, 2);
    auto [msg] = py::parse_pos_or_kw_args<"msg">("rt_assert", args + 1,
                                                 nargs - 1, kwnames);
    auto c = py::ptr(args[0]);
    if (!msg)
        msg = "Assertion failed"_py;
    if (is_rtval(c)) {
        py::ptr seqinfo = self->seqinfo;
        seqinfo->bt_tracker.record(assert_key(seqinfo->assertions.size()));
        seqinfo->assertions.append(py::new_tuple(c, msg));
    }
    else if (!c.as_bool()) {
        py_throw_format(PyExc_AssertionError, "%U", msg);
    }
}

__attribute__((visibility("protected")))
PyTypeObject TimeSeq::Type = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "brassboard_seq.seq.TimeSeq",
    .tp_basicsize = sizeof(TimeSeq),
    .tp_dealloc = seq_dealloc<TimeSeq>,
    .tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE|Py_TPFLAGS_HAVE_GC,
    .tp_traverse = seq_traverse<TimeSeq>,
    .tp_clear = seq_clear<TimeSeq>,
    .tp_methods = (py::meth_table<
                   py::meth_o<"get_channel_id",get_channel_id>,
                   py::meth_fastkw<"set_time",set_time>,
                   py::meth_fastkw<"rt_assert",rt_assert>>),
    .tp_members = (py::mem_table<
                   py::mem_def<"start_time",T_OBJECT_EX,&TimeSeq::start_time,READONLY>,
                   py::mem_def<"end_time",T_OBJECT_EX,&TimeSeq::end_time,READONLY>>),
    .tp_getset = (py::getset_table<
                  py::getset_def<"C",[] (py::ptr<TimeSeq> self) {
                      return py::newref(self->seqinfo->C); }>>),
};

template<bool is_pulse>
inline void TimeStep::set(py::ptr<> chn, py::ptr<> value, py::ptr<> cond,
                          bool exact_time, py::dict_ref &&kws)
{
    int cid;
    if (chn.typeis<py::int_>()) {
        auto lcid = chn.as_int();
        if (lcid < 0 || lcid > seqinfo->channel_paths.size())
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
    io.write_cxx<32>(start_time->data.id);
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
    .tp_dealloc = seq_dealloc<TimeStep>,
    .tp_repr = seq_str<TimeStep>,
    .tp_str = seq_str<TimeStep>,
    .tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_HAVE_GC,
    .tp_traverse = seq_traverse<TimeStep>,
    .tp_clear = seq_clear<TimeStep>,
    .tp_methods = (py::meth_table<
                   py::meth_fastkw<"set",condseq_set<TimeStep,false>>,
                   py::meth_fastkw<"pulse",condseq_set<TimeStep,true>>>),
    .tp_base = &TimeSeq::Type,
};

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
    io.write_cxx<32>(start_time->data.id);
    io.write_ascii("] - T[");
    io.write_cxx<32>(end_time->data.id);
    io.write_ascii("]");
    show_cond_suffix(io);
    show_subseqs(io, indent + 2);
}

template<bool is_pulse>
inline void SubSeq::set(py::ptr<> chn, py::ptr<> value, py::ptr<> cond,
                        bool exact_time, py::dict_ref &&kws)
{
    static_assert(!is_pulse);
    if (dummy_step == Py_None || dummy_step->end_time != end_time) {
        dummy_step.take(add_time_step(cond, end_time, py::int_cached(0)));
        // Update the current time so that a normal step added later
        // this is treated as ordered after this set event
        // rather than at the same time.
        end_time.assign(dummy_step->end_time);
    }
    dummy_step->set<false>(chn, value, cond, exact_time, std::move(kws));
}

inline void SubSeq::wait_cond(py::ptr<> length, py::ptr<> cond)
{
    auto new_time = seqinfo->time_mgr->new_round(end_time, length, cond, Py_None);
    seqinfo->bt_tracker.record(event_time_key(new_time));
    end_time.assign(std::move(new_time));
}

inline void SubSeq::wait_for_cond(py::ptr<> _tp0, py::ptr<> offset, py::ptr<> cond)
{
    py::ptr tp0 = py::cast<EventTime>(_tp0);
    if (!tp0)
        tp0 = py::arg_cast<TimeSeq>(_tp0, "time_point")->end_time;
    auto new_time = seqinfo->time_mgr->new_round(end_time, offset, cond, tp0);
    seqinfo->bt_tracker.record(event_time_key(new_time));
    end_time.assign(std::move(new_time));
}

inline py::ref<SubSeq>
SubSeq::add_custom_step(py::ptr<> cond, py::ptr<EventTime> start_time, py::ptr<> cb,
                        size_t nargs, PyObject *const *args, py::tuple kwnames)
{
    auto subseq = py::generic_alloc<SubSeq>();
    call_constructor(&subseq->seqinfo, py::newref(seqinfo));
    call_constructor(&subseq->start_time, py::newref(start_time));
    call_constructor(&subseq->end_time, py::newref(start_time));
    call_constructor(&subseq->cond, py::newref(cond));
    call_constructor(&subseq->sub_seqs, py::new_list(0));
    call_constructor(&subseq->dummy_step, py::new_none());
    // The python vectorcall ABI allows us to temporarily change the argument array
    // as long as we restore it before returning.
    auto prev_arg = args[-1];
    ((PyObject**)args)[-1] = (PyObject*)subseq;
    ScopeExit restore_arg([&] { ((PyObject**)args)[-1] = prev_arg; });
    cb.vcall(&args[-1], nargs + 1, kwnames);
    sub_seqs.append(subseq);
    return subseq;
}

inline py::ref<TimeStep>
SubSeq::add_time_step(py::ptr<> cond, py::ptr<EventTime> start_time, py::ptr<> length)
{
    auto end_time = seqinfo->time_mgr->new_round(start_time, length, cond, Py_None);
    auto step = py::generic_alloc<TimeStep>();
    call_constructor(&step->actions);
    call_constructor(&step->seqinfo, py::newref(seqinfo));
    call_constructor(&step->start_time, py::newref(start_time));
    call_constructor(&step->end_time, std::move(end_time));
    call_constructor(&step->cond, py::newref(cond));
    call_constructor(&step->length, py::newref(length));
    seqinfo->bt_tracker.record(event_time_key(step->end_time));
    sub_seqs.append(step);
    return step;
}

__attribute__((visibility("protected")))
PyTypeObject SubSeq::Type = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "brassboard_seq.seq.SubSeq",
    .tp_basicsize = sizeof(SubSeq),
    .tp_dealloc = seq_dealloc<SubSeq>,
    .tp_repr = seq_str<SubSeq>,
    .tp_str = seq_str<SubSeq>,
    .tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE|Py_TPFLAGS_HAVE_GC,
    .tp_traverse = seq_traverse<SubSeq>,
    .tp_clear = seq_clear<SubSeq>,
    .tp_methods = (
        py::meth_table<
        py::meth_fastkw<"wait",condseq_wait<SubSeq>>,
        py::meth_fastkw<"wait_for",condseq_wait_for<SubSeq>>,
        py::meth_o<"conditional",condseq_conditional<SubSeq>>,
        py::meth_fastkw<"set",condseq_set<SubSeq>>,
        py::meth_fastkw<"add_step",add_step_real<SubSeq,AddStepType::Step>>,
        py::meth_fastkw<"add_background",add_step_real<SubSeq,AddStepType::Background>>,
        py::meth_fastkw<"add_floating",add_step_real<SubSeq,AddStepType::Floating>>,
        py::meth_fastkw<"add_at",add_step_real<SubSeq,AddStepType::At>>>),
    .tp_members = (py::mem_table<
                   py::mem_def<"current_time",T_OBJECT_EX,&SubSeq::end_time,READONLY>>),
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

__attribute__((visibility("protected")))
PyTypeObject ConditionalWrapper::Type = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "brassboard_seq.seq.ConditionalWrapper",
    .tp_basicsize = sizeof(ConditionalWrapper) + sizeof(void*),
    .tp_dealloc = py::tp_cxx_dealloc<true,ConditionalWrapper>,
    .tp_vectorcall_offset = sizeof(ConditionalWrapper),
    .tp_repr = seq_str<ConditionalWrapper>,
    .tp_call = PyVectorcall_Call,
    .tp_str = seq_str<ConditionalWrapper>,
    .tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_HAVE_GC,
    .tp_traverse = py::tp_field_traverse<ConditionalWrapper,&ConditionalWrapper::seq,
    &ConditionalWrapper::cond>,
    .tp_clear = py::tp_field_clear<ConditionalWrapper,&ConditionalWrapper::seq,
    &ConditionalWrapper::cond>,
    .tp_methods = (
        py::meth_table<
        py::meth_fastkw<"wait",condseq_wait<ConditionalWrapper>>,
        py::meth_fastkw<"wait_for",condseq_wait_for<ConditionalWrapper>>,
        py::meth_o<"conditional",condseq_conditional<ConditionalWrapper>>,
        py::meth_fastkw<"set",condseq_set<ConditionalWrapper>>,
        py::meth_fastkw<"add_step",add_step_real<ConditionalWrapper,AddStepType::Step>>,
        py::meth_fastkw<"add_background",add_step_real<ConditionalWrapper,AddStepType::Background>>,
        py::meth_fastkw<"add_floating",add_step_real<ConditionalWrapper,AddStepType::Floating>>,
        py::meth_fastkw<"add_at",add_step_real<ConditionalWrapper,AddStepType::At>>>),
    .tp_getset = (py::getset_table<
                  py::getset_def<"C",[] (py::ptr<ConditionalWrapper> self) {
                      return py::newref(self->seq->seqinfo->C); }>>),
};

inline void Seq::show(py::stringio &io, int indent) const
{
    io.write_rep_ascii(indent, " ");
    io.write_ascii("Seq - T[");
    io.write_cxx<32>(end_time->data.id);
    io.write_ascii("]\n");
    for (auto [i, t]: py::list_iter(seqinfo->time_mgr->event_times)) {
        io.write_rep_ascii(indent + 1, " ");
        io.write_ascii("T[");
        io.write_cxx<32>(i);
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
    .tp_dealloc = seq_dealloc<Seq>,
    .tp_repr = seq_str<Seq>,
    .tp_str = seq_str<Seq>,
    .tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_HAVE_GC,
    .tp_traverse = seq_traverse<Seq>,
    .tp_clear = seq_clear<Seq>,
    .tp_base = &SubSeq::Type,
    .tp_vectorcall = py::vectorfunc<[] (PyObject*, PyObject *const *args,
                                        ssize_t nargs, py::tuple kwnames) {
        py::check_num_arg("Seq.__init__", nargs, 1, 2);
        auto [py_max_frame] =
            py::parse_pos_or_kw_args<"max_frame">("Seq.__init__", args + 1,
                                                  nargs - 1, kwnames);
        int max_frame = 0;
        if (py_max_frame) {
            max_frame = py_max_frame.as_int();
            if (max_frame < 0) {
                py_throw_format(PyExc_ValueError, "max_frame cannot be negative");
            }
        }
        auto self = py::generic_alloc<Seq>();
        call_constructor(&self->start_time, py::new_none());
        call_constructor(&self->cond, py::new_true());
        call_constructor(&self->sub_seqs, py::new_list(0));
        call_constructor(&self->dummy_step, py::new_none());
        auto seqinfo = py::generic_alloc<SeqInfo>();
        call_constructor(&seqinfo->bt_tracker);
        seqinfo->bt_tracker.max_frame = max_frame;
        call_constructor(&seqinfo->action_alloc);
        seqinfo->action_counter = 0;
        call_constructor(&seqinfo->config, py::newref(py::arg_cast<config::Config>(args[0], "config")));
        call_constructor(&seqinfo->time_mgr, event_time::TimeManager::alloc());
        call_constructor(&seqinfo->assertions, py::new_list(0));
        call_constructor(&seqinfo->channel_name_map, py::new_dict());
        call_constructor(&seqinfo->channel_path_map, py::new_dict());
        call_constructor(&seqinfo->channel_paths, py::new_list(0));
        call_constructor(&seqinfo->C, scan::ParamPack::new_empty());
        call_constructor(&self->end_time,
                         seqinfo->time_mgr->new_int(Py_None, 0, false,
                                                    Py_True, Py_None));
        seqinfo->bt_tracker.record(event_time_key(self->end_time));
        call_constructor(&self->seqinfo, std::move(seqinfo));
        return self;
    }>,
};

__attribute__((visibility("hidden")))
void init()
{
    throw_if(PyType_Ready(&SeqInfo::Type) < 0);
    throw_if(PyType_Ready(&TimeSeq::Type) < 0);
    throw_if(PyType_Ready(&TimeStep::Type) < 0);
    throw_if(PyType_Ready(&SubSeq::Type) < 0);
    throw_if(PyType_Ready(&ConditionalWrapper::Type) < 0);
    throw_if(PyType_Ready(&Seq::Type) < 0);
}

}
