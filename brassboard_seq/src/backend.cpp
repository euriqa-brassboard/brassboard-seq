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

#include "seq.h"

#include <algorithm>
#include <vector>

namespace brassboard_seq::backend {

using namespace rtval;
using seq::TimeStep;
using seq::SubSeq;
using seq::BasicSeq;
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
inline void CompiledSeq::populate_bseq_values(py::ptr<seq::Seq> seq,
                                              CompiledBasicSeq *cbseq,
                                              std::vector<uint8_t> &chn_status)
{
    py::ptr ginfo = seq->seqinfo;
    auto cinfo = ginfo->cinfo.get();
    auto chn_actions = cbseq->chn_actions.get();
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
    py::ptr basic_seqs = seq->basic_seqs;
    auto bseq = basic_seqs.get<BasicSeq>(cbseq->bseq_id);
    for (auto next_id: bseq->next_bseq) {
        auto next_cbseq = basic_cseqs[next_id];
        auto next_chn_actions = next_cbseq->chn_actions.get();
        if (!next_cbseq->chn_actions[0]->start_value) {
            for (int chn = 0; chn < nchn; chn++)
                next_chn_actions[chn]->start_value.assign(final_values[chn]);
            std::ranges::fill(chn_status, false);
            populate_bseq_values(seq, next_cbseq, chn_status);
            cbseq->next_bseq.push_back(next_id);
            continue;
        }
        auto &new_cbseq = *basic_seq_alloc.alloc();
        auto new_cbseq_id = (int)basic_cseqs.size();
        basic_cseqs.push_back(&new_cbseq);
        new_cbseq.bseq_id = next_id;
        new_cbseq.may_term = next_cbseq->may_term;
        cbseq->next_bseq.push_back(new_cbseq_id);
        auto new_chn_actions = new ChannelAction*[nchn];
        new_cbseq.chn_actions.reset(new_chn_actions);
        for (int chn = 0; chn < nchn; chn++) {
            bool found = false;
            auto &action_list = get_action_list(chn, next_id);
            for (auto chn_action: action_list) {
                if (same_value(chn_action->start_value, final_values[chn])) {
                    new_chn_actions[chn] = chn_action;
                    found = true;
                    break;
                }
            }
            chn_status[chn] = found;
            if (found)
                continue;
            auto chn_action = new_chn_action(chn, next_id);
            new_chn_actions[chn] = chn_action;
            chn_action->start_value.assign(final_values[chn]);
            chn_action->actions = action_list[0]->actions;
        }
        populate_bseq_values(seq, &new_cbseq, chn_status);
    }
}

static inline auto basic_seq_key(py::ptr<BasicSeq> bseq)
{
    return event_time_key(bseq->seqinfo->time_mgr->event_times.get(0));
}

static void _check_seq_flow(py::ptr<BasicSeq> bseq, std::vector<uint8_t> &visit_status)
{
    auto &status = visit_status[bseq->bseq_id];
    if (status == 2)
        return;
    if (status == 1)
        bb_throw_format(PyExc_ValueError, basic_seq_key(bseq), "Loop found in sequence");
    status = 1;
    for (auto next_id: bseq->next_bseq)
        _check_seq_flow(bseq->basic_seqs.get<BasicSeq>(next_id), visit_status);
    status = 2;
}

static inline void check_seq_flow(py::ptr<BasicSeq> bseq)
{
    std::vector<uint8_t> status(bseq->basic_seqs.size(), 0);
    _check_seq_flow(bseq, status);
    for (auto [i, s]: py::list_iter<BasicSeq>(bseq->basic_seqs)) {
        if (status[i] != 2) {
            bb_throw_format(PyExc_ValueError, basic_seq_key(s),
                            "BasicSeq %d unreachable", i);
        }
    }
}

__attribute__((visibility("internal")))
inline void CompiledSeq::initialize(py::ptr<seq::Seq> seq)
{
    py::ptr ginfo = seq->seqinfo;
    nchn = ginfo->channel_paths.size();
    nbseq = seq->basic_seqs.size();
    all_chn_actions.resize(nchn * nbseq);
    basic_cseqs.resize(nbseq);
    for (auto [bseq_id, bseq]: py::list_iter<BasicSeq>(seq->basic_seqs)) {
        assert(bseq->bseq_id == bseq_id);
        py::ptr binfo = bseq->seqinfo;
        py::ptr time_mgr = binfo->time_mgr;
        time_mgr->finalize();
        auto &cbseq = *basic_seq_alloc.alloc();
        basic_cseqs[bseq_id] = &cbseq;
        cbseq.bseq_id = bseq_id;
        cbseq.may_term = bseq->may_terminate();
        auto chn_actions = new ChannelAction*[nchn];
        cbseq.chn_actions.reset(chn_actions);
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
inline void CompiledSeq::populate_values(py::ptr<seq::Seq> seq)
{
    if (nchn == 0) [[unlikely]]
        return;
    auto chn_actions0 = basic_cseqs[0]->chn_actions.get();
    for (int chn = 0; chn < nchn; chn++)
        chn_actions0[chn]->start_value.take(py::new_false());
    std::vector<uint8_t> chn_status(nchn, false);
    populate_bseq_values(seq, basic_cseqs[0], chn_status);
}

template<typename Backend>
static inline void compiler_finalize(auto comp, Backend*)
{
    py::ptr seq = comp->seq;
    py::ptr seqinfo = seq->seqinfo;
    auto bt_guard = set_global_tracker(&seqinfo->cinfo->bt_tracker);
    // This relies on the event times being in the order of allocation
    // so this should be done before finalizing the time manager.
    check_seq_flow(seq);
    for (auto [i, path]: py::list_iter<py::tuple>(seqinfo->channel_paths)) {
        auto prefix = path.get(0);
        if (py::dict(comp->backends).contains(prefix)) [[likely]]
            continue;
        py_throw_format(PyExc_ValueError, "Unhandled channel: %U",
                        config::channel_name_from_path(path));
    }
    seqinfo->channel_name_map.take(py::new_none()); // Free up memory
    auto &cseq = comp->cseq;
    cseq.initialize(seq);
    cseq.populate_values(seq);
    for (auto [name, backend]: py::dict_iter<Backend>(comp->backends)) {
        throw_if(backend->__pyx_vtab->finalize(backend, cseq) < 0);
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
inline void CompiledSeq::eval_chn_actions(py::ptr<seq::Seq> seq, unsigned age)
{
    int ncbseq = basic_cseqs.size();
    py::ptr basic_seqs = seq->basic_seqs;
    for (int cbseq_id = 0; cbseq_id < ncbseq; cbseq_id++) {
        auto cbseq = basic_cseqs[cbseq_id];
        auto bseq_id = cbseq->bseq_id;
        auto bseq = basic_seqs.get<BasicSeq>(bseq_id);
        if (bseq_id != cbseq_id) {
            // We handle all the actions when handling the original cbseq
            cbseq->total_time = basic_cseqs[bseq_id]->total_time;
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

template<typename Backend>
static inline void compiler_runtime_finalize(auto comp, py::ptr<> _age, Backend*)
{
    unsigned age = _age.as_int();
    py::ptr seq = comp->seq;
    py::ptr ginfo = seq->seqinfo;
    auto &cseq = comp->cseq;
    auto bt_guard = set_global_tracker(&ginfo->cinfo->bt_tracker);
    for (auto [bseq_id, bseq]: py::list_iter<BasicSeq>(seq->basic_seqs)) {
        assert(cseq.basic_cseqs[bseq_id]->bseq_id == bseq_id);
        cseq.basic_cseqs[bseq_id]->total_time =
            bseq->seqinfo->time_mgr->compute_all_times(age);
    }
    for (auto [assert_id, a]: py::list_iter<py::tuple>(ginfo->assertions)) {
        auto c = a.get(0);
        rt_eval_throw(c, age, assert_key(assert_id));
        if (rtval_cache(c).is_zero()) {
            bb_throw_format(PyExc_AssertionError, assert_key(assert_id), "%U", a.get(1));
        }
    }
    cseq.eval_chn_actions(seq, age);
    for (auto [name, backend]: py::dict_iter<Backend>(comp->backends)) {
        throw_if(backend->__pyx_vtab->runtime_finalize(backend, cseq, age) < 0);
    }
}

}
