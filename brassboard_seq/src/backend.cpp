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

#include "backend.h"

#include <algorithm>
#include <vector>

namespace brassboard_seq::backend {

using namespace rtval;
using seq::TimeStep;
using seq::SubSeq;
using event_time::EventTime;

static void collect_actions(SubSeq *self, ChannelAction **chn_actions)
{
    for (auto [i, subseq]: py::list_iter(self->sub_seqs)) {
        auto step = py::cast<TimeStep,true>(subseq);
        if (!step) {
            collect_actions((SubSeq*)subseq, chn_actions);
            continue;
        }
        auto tid = step->start_time->data.id;
        auto end_tid = step->end_time->data.id;
        int nactions = step->actions.size();
        for (int chn = 0; chn < nactions; chn++) {
            auto action = step->actions[chn];
            if (!action)
                continue;
            action->tid = tid;
            action->end_tid = end_tid;
            chn_actions[chn]->actions.push_back(action);
        }
    }
}

__attribute__((visibility("internal")))
inline void SeqCompiler::populate_bseq_values(CompiledBasicSeq &cbseq,
                                              std::vector<uint8_t> &chn_status)
{
    py::ptr ginfo = seq->seqinfo;
    auto cinfo = ginfo->cinfo.get();
    auto chn_actions = cbseq.chn_actions.get();
    std::vector<py::ref<>> final_values(nchn);
    for (int chn = 0; chn < nchn; chn++) {
        auto chn_action = chn_actions[chn];
        auto &actions = chn_action->actions;
        py::ref value(chn_action->start_value.ref());
        if (actions.empty()) {
            final_values[chn].take(std::move(value));
            continue;
        }
        else if (chn_status[chn]) {
            final_values[chn].assign(actions.back()->end_val);
            continue;
        }
        for (auto &_action: actions) {
            auto action = _action;
            py::ptr action_value = action->value;
            auto isramp = action::isramp(action_value);
            py::ptr cond = action->cond;
            bool initialized(action->end_val);
            if (!action->is_pulse && cond != Py_False) {
                py::ref<> new_value;
                if (isramp) {
                    auto rampf = (action::RampFunctionBase*)action_value;
                    try {
                        new_value.take(rampf->eval_end(action->length, value));
                    }
                    catch (...) {
                        bb_rethrow(action_key(action->aid));
                    }
                }
                else if (initialized) {
                    // For non-ramp the end value does not depend on the
                    // old value so we can skip all the actions afterwards
                    break;
                }
                else {
                    new_value = action_value.ref();
                }
                if (cond == Py_True) {
                    value = std::move(new_value);
                }
                else if (new_value != value) {
                    assert(is_rtval(cond));
                    value = new_select(cond, new_value, value);
                }
                if (initialized && same_value(value, action->end_val)) {
                    break;
                }
            }
            if (!initialized) {
                action->end_val.assign(value);
            }
            else {
                auto new_action = cinfo->action_alloc.alloc(
                    action_value, cond, action->is_pulse, action->exact_time,
                    action->kws.xref(), action->aid);
                new_action->tid = action->tid;
                new_action->end_tid = action->end_tid;
                new_action->length = action->length;
                new_action->end_val.assign(value);
                _action = new_action;
            }
        }
        final_values[chn].assign(actions.back()->end_val);
    }
    for (auto next_cbseq_id: cbseq.next_bseq) {
        auto &next_cbseq = basic_cseqs[next_cbseq_id];
        auto next_chn_actions = next_cbseq.chn_actions.get();
        auto next_bseq_id = next_cbseq.bseq_id;
        if (next_bseq_id == next_cbseq_id) {
            for (int chn = 0; chn < nchn; chn++)
                next_chn_actions[chn]->start_value.assign(final_values[chn]);
            std::ranges::fill(chn_status, false);
        }
        else {
            for (int chn = 0; chn < nchn; chn++) {
                bool found = false;
                auto &action_list = get_action_list(chn, next_bseq_id);
                for (auto chn_action: action_list) {
                    if (same_value(chn_action->start_value, final_values[chn])) {
                        next_chn_actions[chn] = chn_action;
                        found = true;
                        break;
                    }
                }
                chn_status[chn] = found;
                if (found)
                    continue;
                auto chn_action = new_chn_action(chn, next_bseq_id);
                next_chn_actions[chn] = chn_action;
                chn_action->start_value.assign(final_values[chn]);
                chn_action->actions = action_list[0]->actions;
            }
        }
        populate_bseq_values(next_cbseq, chn_status);
    }
}

static inline auto basic_seq_key(py::ptr<BasicSeq> bseq)
{
    return event_time_key(bseq->seqinfo->time_mgr->event_times.get(0));
}

__attribute__((visibility("internal")))
inline int SeqCompiler::visit_bseq(py::ptr<BasicSeq> bseq,
                                   std::vector<uint8_t> &visit_status)
{
    auto bseq_id = bseq->bseq_id;
    auto &status = visit_status[bseq_id];
    if (status == 1)
        bb_throw_format(PyExc_ValueError, basic_seq_key(bseq), "Loop found in sequence");
    int cbseq_id;
    if (status == 2) {
        cbseq_id = basic_cseqs.size();
        basic_cseqs.emplace_back();
    }
    else {
        status = 1;
        cbseq_id = bseq->bseq_id;
    }
    std::vector<int> next_bseq;
    for (auto next_id: bseq->next_bseq)
        next_bseq.push_back(visit_bseq(bseq->basic_seqs.get<BasicSeq>(next_id),
                                       visit_status));
    auto &cbseq = basic_cseqs[cbseq_id];
    cbseq.bseq_id = bseq_id;
    cbseq.may_term = bseq->may_terminate();
    cbseq.next_bseq = std::move(next_bseq);
    cbseq.chn_actions.reset(new ChannelAction*[nchn]);
    status = 2;
    return cbseq_id;
}

__attribute__((visibility("internal")))
inline void SeqCompiler::initialize_bseqs()
{
    py::ptr ginfo = seq->seqinfo;
    nchn = ginfo->channel_paths.size();
    nbseq = seq->basic_seqs.size();
    std::vector<uint8_t> status(nbseq, 0);
    basic_cseqs.resize(nbseq);
    visit_bseq(seq, status);
    for (auto [i, s]: py::list_iter<BasicSeq>(seq->basic_seqs)) {
        if (status[i] != 2) {
            bb_throw_format(PyExc_ValueError, basic_seq_key(s),
                            "BasicSeq %d unreachable", i);
        }
    }
}

__attribute__((visibility("internal")))
inline void SeqCompiler::initialize_actions()
{
    all_chn_actions.resize(nchn * nbseq);
    for (auto [bseq_id, bseq]: py::list_iter<BasicSeq>(seq->basic_seqs)) {
        assert(bseq->bseq_id == bseq_id);
        py::ptr binfo = bseq->seqinfo;
        py::ptr time_mgr = binfo->time_mgr;
        time_mgr->finalize();
        auto &cbseq = basic_cseqs[bseq_id];
        assert(cbseq.bseq_id == bseq_id);
        auto chn_actions = cbseq.chn_actions.get();
        for (int chn = 0; chn < nchn; chn++)
            chn_actions[chn] = new_chn_action(chn, bseq_id);
        collect_actions(bseq, chn_actions);
        auto get_time = [event_times=time_mgr->event_times.ptr()] (int tid) {
            return event_times.get<EventTime>(tid);
        };
        for (int cid = 0; cid < nchn; cid++) {
            auto chn_action = chn_actions[cid];
            std::ranges::sort(chn_action->actions, [] (auto *a1, auto *a2) {
                return a1->tid < a2->tid;
            });
            EventTime *last_time = nullptr;
            bool last_is_start = false;
            int tid = -1;
            for (auto action: chn_action->actions) {
                // It is difficult to decide the ordering of actions
                // if multiple were added to exactly the same time points.
                // We disallow this in the same timestep and we'll also disallow
                // this here.
                if (action->tid == tid)
                    bb_throw_format(PyExc_ValueError, action_key(action->aid),
                                    "Multiple actions added for the same channel "
                                    "at the same time on %U.",
                                    binfo->channel_name_from_id(cid));
                tid = action->tid;
                auto start_time = get_time(tid);
                if (last_time) {
                    auto o = event_time::is_ordered(last_time, start_time);
                    if (o != event_time::OrderBefore &&
                        (o != event_time::OrderEqual || last_is_start)) {
                        bb_throw_format(PyExc_ValueError, action_key(action->aid),
                                        "Actions on %U is not statically ordered",
                                        binfo->channel_name_from_id(cid));
                    }
                }
                py::ptr action_value = action->value;
                auto isramp = action::isramp(action_value);
                last_is_start = !(action->is_pulse || isramp);
                last_time = last_is_start ? start_time : get_time(action->end_tid);
            }
        }
    }
}

__attribute__((visibility("internal")))
inline void SeqCompiler::populate_values()
{
    auto chn_actions0 = basic_cseqs[0].chn_actions.get();
    for (int chn = 0; chn < nchn; chn++)
        chn_actions0[chn]->start_value.take(py::new_false());
    std::vector<uint8_t> chn_status(nchn, false);
    populate_bseq_values(basic_cseqs[0], chn_status);
}

__attribute__((visibility("internal")))
inline void SeqCompiler::finalize()
{
    py::ptr seqinfo = seq->seqinfo;
    auto bt_guard = set_global_tracker(&seqinfo->cinfo->bt_tracker);
    // This relies on the event times being in the order of allocation
    // so this should be done before finalizing the time manager.
    initialize_bseqs();
    for (auto [i, path]: py::list_iter<py::tuple>(seqinfo->channel_paths)) {
        auto prefix = path.get(0);
        if (py::dict(backends).contains(prefix)) [[likely]]
            continue;
        py_throw_format(PyExc_ValueError, "Unhandled channel: %U",
                        config::channel_name_from_path(path));
    }
    seqinfo->channel_name_map.take(py::new_none()); // Free up memory
    initialize_actions();
    populate_values();
    for (auto [name, backend]: py::dict_iter<BackendBase>(backends)) {
        backend->data()->finalize(this);
    }
}

static inline auto action_get_condval(auto action, unsigned age)
{
    py::ptr cond = action->cond;
    if (cond == Py_True)
        return true;
    if (cond == Py_False)
        return false;
    assert(is_rtval(cond));
    try {
        rt_eval_throw(cond, age);
        return !rtval_cache(cond).is_zero();
    }
    catch (...) {
        bb_rethrow(action_key(action->aid));
    }
}

__attribute__((visibility("internal")))
inline void SeqCompiler::eval_chn_actions(unsigned age)
{
    int ncbseq = basic_cseqs.size();
    py::ptr basic_seqs = seq->basic_seqs;
    for (int cbseq_id = 0; cbseq_id < ncbseq; cbseq_id++) {
        auto &cbseq = basic_cseqs[cbseq_id];
        auto bseq_id = cbseq.bseq_id;
        auto bseq = basic_seqs.get<BasicSeq>(bseq_id);
        if (bseq_id != cbseq_id) {
            // We handle all the actions when handling the original cbseq
            cbseq.total_time = basic_cseqs[bseq_id].total_time;
            continue;
        }
        py::ptr time_mgr = bseq->seqinfo->time_mgr;
        for (int chn = 0; chn < nchn; chn++) {
            auto chn_actions = get_action_list(chn, bseq_id);
            bool first = true;
            for (auto chn_action: chn_actions) {
                auto &actions = chn_action->actions;
                bool check_time = std::exchange(first, false);
                int64_t prev_time = 0;
                for (auto action: actions) {
                    bool cond_val = action_get_condval(action, age);
                    action->cond_val = cond_val;
                    if (!cond_val)
                        continue;
                    py::ptr action_value = action->value;
                    auto isramp = action::isramp(action_value);
                    if (isramp) {
                        auto rampf = (action::RampFunctionBase*)action_value;
                        try {
                            rampf->set_runtime_params(age);
                        }
                        catch (...) {
                            bb_rethrow(action_key(action->aid));
                        }
                    }
                    else if (is_rtval(action_value)) {
                        rt_eval_throw(action_value, age, action_key(action->aid));
                    }
                    // No need to evaluate action.length since the `compute_all_times`
                    // above should've done it already.
                    py::ptr action_end_val = action->end_val;
                    if (action_end_val != action_value && is_rtval(action_end_val))
                        rt_eval_throw(action_end_val, age, action_key(action->aid));
                    // Currently all versions of channel action should share
                    // the same time points. We only need to check the time ordering once.
                    if (!check_time)
                        continue;
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
}

__attribute__((visibility("internal")))
inline void SeqCompiler::runtime_finalize(py::ptr<> _age)
{
    unsigned age = _age.as_int();
    py::ptr ginfo = seq->seqinfo;
    auto bt_guard = set_global_tracker(&ginfo->cinfo->bt_tracker);
    for (auto [bseq_id, bseq]: py::list_iter<BasicSeq>(seq->basic_seqs)) {
        assert(basic_cseqs[bseq_id].bseq_id == bseq_id);
        basic_cseqs[bseq_id].total_time =
            bseq->seqinfo->time_mgr->compute_all_times(age);
    }
    for (auto [assert_id, a]: py::list_iter<py::tuple>(ginfo->assertions)) {
        auto c = a.get(0);
        rt_eval_throw(c, age, assert_key(assert_id));
        if (rtval_cache(c).is_zero()) {
            bb_throw_format(PyExc_AssertionError, assert_key(assert_id), "%U", a.get(1));
        }
    }
    eval_chn_actions(age);
    for (auto [name, backend]: py::dict_iter<BackendBase>(backends)) {
        backend->data()->runtime_finalize(this, age);
    }
}

__attribute__((visibility("hidden")))
PyTypeObject BackendBase::Type = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "brassboard_seq.backend.BackendBase",
    .tp_basicsize = sizeof(BackendBase) + sizeof(BackendBase::Data),
    .tp_flags = Py_TPFLAGS_DEFAULT
};

__attribute__((visibility("protected")))
PyTypeObject SeqCompiler::Type = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "brassboard_seq.backend.SeqCompiler",
    .tp_basicsize = sizeof(SeqCompiler),
    .tp_dealloc = py::tp_cxx_dealloc<true,SeqCompiler>,
    .tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE|Py_TPFLAGS_HAVE_GC,
    .tp_traverse = py::tp_field_traverse<SeqCompiler,&SeqCompiler::seq,&SeqCompiler::backends>,
    .tp_clear = py::tp_field_clear<SeqCompiler,&SeqCompiler::seq,&SeqCompiler::backends>,
    .tp_methods = (
        py::meth_table<
        py::meth_fast<"add_backend",[] (py::ptr<SeqCompiler> self, PyObject *const *args,
                                        Py_ssize_t nargs) {
            py::check_num_arg("SeqCompiler.add_backend", nargs, 2, 2);
            auto name = py::arg_cast<py::str>(args[0], "name");
            auto backend = py::arg_cast<BackendBase>(args[1], "backend");
            if (self->backends.contains(name))
                py_throw_format(PyExc_ValueError, "Backend %U already exist", name);
            self->backends.set(name, backend);
            backend->data()->prefix.assign(name);
        }>,
        py::meth_noargs<"finalize",[] (py::ptr<SeqCompiler> self) { self->finalize(); }>,
        py::meth_o<"runtime_finalize",[] (py::ptr<SeqCompiler> self, py::ptr<> pyage) {
            self->runtime_finalize(pyage);
        }>>),
    .tp_members = (py::mem_table<
                   py::mem_def<"seq",T_OBJECT_EX,&SeqCompiler::seq,READONLY>>),
    .tp_init = py::itrifunc<[] (py::ptr<SeqCompiler> self, py::tuple args, py::dict kws) {
        py::check_num_arg("SeqCompiler.__init__", args ? args.size() : 0, 1, 1);
        if (kws) {
            for (auto [name, _]: py::dict_iter(kws)) {
                unexpected_kwarg_error("SeqCompiler.__init__", name);
            }
        }
        self->seq.assign(py::arg_cast<seq::Seq>(args.get(0), "seq"));
    }>,
    .tp_new = py::tp_new<[] (PyTypeObject *t, auto...) {
        auto self = py::generic_alloc<SeqCompiler>(t);
        call_constructor(&self->seq, py::new_none());
        call_constructor(&self->backends, py::new_dict());
        return self;
    }>,
};

namespace {
struct Backend : BackendBase::Base<Backend> {
    struct Data final : BackendBase::Data {};
    static PyTypeObject Type;
};
PyTypeObject Backend::Type = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "brassboard_seq.backend.Backend",
    .tp_basicsize = sizeof(BackendBase) + sizeof(Backend::Data),
    .tp_dealloc = py::tp_cxx_dealloc<false,Backend>,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_base = &BackendBase::Type,
    .tp_vectorcall = py::vectorfunc<[] (auto, PyObject *const *args,
                                        ssize_t nargs, py::tuple kwnames) {
        py::check_num_arg("Backend.__init__", nargs, 0, 0);
        py::check_no_kwnames("Backend.__init__", kwnames);
        auto self = py::generic_alloc<Backend>();
        call_constructor(self->data());
        return self;
    }>
};
} // (anonymous)

PyTypeObject &Backend_Type = Backend::Type;

__attribute__((visibility("hidden")))
void init()
{
    throw_if(PyType_Ready(&BackendBase::Type) < 0);
    throw_if(PyType_Ready(&Backend::Type) < 0);
    throw_if(PyType_Ready(&SeqCompiler::Type) < 0);
}

}
