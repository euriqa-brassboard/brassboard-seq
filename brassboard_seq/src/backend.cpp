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

#include "Python.h"

#include "event_time.h"
#include "seq.h"

#include <algorithm>
#include <vector>

namespace brassboard_seq::backend {

static PyObject *timestep_type;
static PyObject *rampfunctionbase_type;

using namespace rtval;

template<typename TimeStep, typename SubSeq>
static void collect_actions(SubSeq *self, std::vector<action::Action*> *actions)
{
    for (auto [i, subseq]: pylist_iter(self->sub_seqs)) {
        if (Py_TYPE(subseq) != (PyTypeObject*)timestep_type) {
            collect_actions<TimeStep>((SubSeq*)subseq, actions);
            continue;
        }
        auto step = (TimeStep*)subseq;
        auto tid = pyx_fld(step, start_time)->data.id;
        auto end_tid = pyx_fld(step, end_time)->data.id;
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

static inline bool check_compiled_bseq(CompiledBasicSeq *cbseq, int bseq_id,
                                       const std::vector<py_object> &start_values)
{
    if (cbseq->bseq_id != bseq_id)
        return false;
    auto nchn = start_values.size();
    for (size_t i = 0; i < nchn; i++)
        if (!rt_same_value(cbseq->start_values[i].get(), start_values[i].get()))
            return false;
    return true;
}

template<typename _RampFunctionBase>
static inline void populate_bseq_values(auto comp, CompiledBasicSeq *cbseq)
{
    auto seqinfo = pyx_fld(comp->seq, seqinfo);
    auto cinfo = seqinfo->cinfo;
    auto basic_seqs = pyx_fld(comp->seq, basic_seqs);
    using BasicSeq = std::remove_reference_t<decltype(*pyx_find_base(comp->seq, bseq_id))>;
    auto bseq = (BasicSeq*)PyList_GET_ITEM(basic_seqs, cbseq->bseq_id);
    auto nchn = (int)PyList_GET_SIZE(seqinfo->channel_paths);
    auto all_actions = cbseq->all_actions.get();
    std::vector<py_object> final_values(nchn);
    for (int cid = 0; cid < nchn; cid++) {
        auto &actions = all_actions[cid];
        py_object value(py_newref(cbseq->start_values[cid].get()));
        for (auto &_action: actions) {
            auto action = _action;
            auto action_value = action->value.get();
            auto isramp = py_issubtype_nontrivial(Py_TYPE(action_value),
                                                  rampfunctionbase_type);
            auto cond = action->cond.get();
            bool initialized = action->end_val;
            if (!action->is_pulse) {
                if (cond != Py_False) {
                    py_object new_value;
                    if (isramp) {
                        auto rampf = (_RampFunctionBase*)action_value;
                        auto length = action->length;
                        auto vt = rampf->__pyx_vtab;
                        new_value.reset_checked(vt->eval_end(rampf, length, value),
                                                action_key(action->aid));
                    }
                    else if (initialized) {
                        // For non-ramp the end value does not depend on the
                        // old value so we can skip all the actions afterwards
                        break;
                    }
                    else {
                        new_value.set_obj(action_value);
                    }
                    if (cond == Py_True) {
                        std::swap(value, new_value);
                    }
                    else if (new_value.get() != value.get()) {
                        assert(is_rtval(cond));
                        value.reset(new_select(cond, new_value, value));
                    }
                }
            }
            if (!initialized) {
                action->end_val.set_obj(value.get());
            }
            else if (rt_same_value(value, action->end_val)) {
                break;
            }
            else {
                auto new_action = cinfo->action_alloc.alloc(
                    action_value, cond, action->is_pulse, action->exact_time,
                    py_object(py_newref(action->kws.get())), action->aid);
                new_action->tid = action->tid;
                new_action->end_tid = action->end_tid;
                new_action->length = action->length;
                new_action->end_val.set_obj(value.get());
            }
        }
        final_values[cid].set_obj(actions.empty() ? value.get() :
                                  actions.back()->end_val.get());
    }
    auto nbseqs = (int)PyList_GET_SIZE(basic_seqs);
    for (auto next_id: pyx_fld(bseq, next_bseq)) {
        auto next_cbseq = comp->cseq.basic_seqs[next_id];
        if (!next_cbseq->start_values[0]) {
            for (int i = 0; i < nchn; i++)
                next_cbseq->start_values[i].set_obj(final_values[i].get());
            populate_bseq_values<_RampFunctionBase>(comp, next_cbseq);
            cbseq->next_bseq.push_back(next_id);
            continue;
        }
        if (check_compiled_bseq(next_cbseq, next_id, final_values)) {
            cbseq->next_bseq.push_back(next_id);
            continue;
        }
        auto nbcseqs = (int)comp->cseq.basic_seqs.size();
        bool found = false;
        for (int i = nbseqs; i < nbcseqs; i++) {
            if (check_compiled_bseq(comp->cseq.basic_seqs[i], next_id, final_values)) {
                cbseq->next_bseq.push_back(i);
                found = true;
                break;
            }
        }
        if (found)
            continue;
        auto &new_cbseq = comp->cseq.basic_seq_list.emplace_back();
        comp->cseq.basic_seqs.push_back(&new_cbseq);
        new_cbseq.bseq_id = next_id;
        new_cbseq.may_term = next_cbseq->may_term;
        cbseq->next_bseq.push_back(nbcseqs);
        auto new_all_actions = new std::vector<action::Action*>[nchn];
        new_cbseq.all_actions.reset(new_all_actions);
        new_cbseq.start_values.resize(nchn);
        for (int i = 0; i < nchn; i++) {
            new_cbseq.start_values[i].set_obj(final_values[i].get());
            new_all_actions[i] = next_cbseq->all_actions[i];
        }
        populate_bseq_values<_RampFunctionBase>(comp, &new_cbseq);
    }
}

static inline auto basic_seq_key(auto *bseq)
{
    auto event_times = pyx_fld(bseq, seqinfo)->time_mgr->event_times;
    return event_time_key(PyList_GET_ITEM(event_times, 0));
}

template<typename BasicSeq>
static void _check_seq_flow(BasicSeq *bseq, std::vector<uint8_t> &visit_status)
{
    auto &status = visit_status[pyx_fld(bseq, bseq_id)];
    if (status == 2)
        return;
    if (status == 1)
        bb_throw_format(PyExc_ValueError, basic_seq_key(bseq),
                        "Loop found in sequence");
    status = 1;
    for (auto next_id: pyx_fld(bseq, next_bseq))
        _check_seq_flow((BasicSeq*)PyList_GET_ITEM(pyx_fld(bseq, basic_seqs), next_id),
                        visit_status);
    status = 2;
}

template<typename BasicSeq>
static inline void check_seq_flow(BasicSeq *bseq)
{
    auto nbseq = (int)PyList_GET_SIZE(pyx_fld(bseq, basic_seqs));
    std::vector<uint8_t> status(nbseq, 0);
    _check_seq_flow(bseq, status);
    for (auto [i, s]: pylist_iter<BasicSeq>(pyx_fld(bseq, basic_seqs))) {
        if (status[i] != 2) {
            bb_throw_format(PyExc_ValueError, basic_seq_key(s),
                            "BasicSeq %d unreachable", i);
        }
    }
}

template<typename TimeStep, typename _RampFunctionBase, typename Backend>
static inline void compiler_finalize(auto comp, TimeStep*, _RampFunctionBase*, Backend*)
{
    auto seq = comp->seq;
    using EventTime = std::remove_reference_t<decltype(*pyx_fld(seq, end_time))>;
    using BasicSeq = std::remove_reference_t<decltype(*pyx_find_base(seq, bseq_id))>;
    auto seqinfo = pyx_fld(seq, seqinfo);
    auto bt_guard = set_global_tracker(&seqinfo->cinfo->bt_tracker);
    // This relies on the event times being in the order of allocation
    // so this should be done before finalizing the time manager.
    check_seq_flow((BasicSeq*)seq);
    auto nchn = (int)PyList_GET_SIZE(seqinfo->channel_paths);
    for (auto [i, path]: pylist_iter(seqinfo->channel_paths)) {
        auto prefix = PyTuple_GET_ITEM(path, 0);
        auto res = PyDict_Contains(comp->backends, prefix);
        if (res > 0) [[likely]]
            continue;
        throw_if(res < 0);
        py_throw_format(PyExc_ValueError, "Unhandled channel: %U",
                        channel_name_from_path(path).get());
    }
    pyassign(seqinfo->channel_name_map, Py_None); // Free up memory
    auto &cseq = comp->cseq;
    cseq.basic_seqs.resize(PyList_GET_SIZE(pyx_fld(seq, basic_seqs)));
    for (auto [i, bseq]: pylist_iter<BasicSeq>(pyx_fld(seq, basic_seqs))) {
        assert(bseq->bseq_id == i);
        auto seqinfo = pyx_fld(bseq, seqinfo);
        auto time_mgr = seqinfo->time_mgr;
        time_mgr->__pyx_vtab->finalize(time_mgr);
        auto &cbseq = cseq.basic_seq_list.emplace_back();
        cseq.basic_seqs[i] = &cbseq;
        cbseq.bseq_id = i;
        cbseq.may_term = seq::basicseq_may_terminate(bseq);
        auto all_actions = new std::vector<action::Action*>[nchn];
        cbseq.all_actions.reset(all_actions);
        cbseq.start_values.resize(nchn);
        collect_actions<TimeStep>(pyx_find_base(bseq, sub_seqs), all_actions);
        auto get_time = [event_times=time_mgr->event_times] (int tid) {
            return (EventTime*)PyList_GET_ITEM(event_times, tid);
        };
        for (int cid = 0; cid < nchn; cid++) {
            auto &actions = all_actions[cid];
            std::ranges::sort(actions, [] (auto *a1, auto *a2) {
                return a1->tid < a2->tid;
            });
            EventTime *last_time = nullptr;
            bool last_is_start = false;
            int tid = -1;
            for (auto action: actions) {
                // It is difficult to decide the ordering of actions
                // if multiple were added to exactly the same time points.
                // We disallow this in the same timestep and we'll also disallow
                // this here.
                if (action->tid == tid)
                    bb_throw_format(PyExc_ValueError, action_key(action->aid),
                                    "Multiple actions added for the same channel "
                                    "at the same time on %U.",
                                    seq::channel_name_from_id(seqinfo, cid).get());
                tid = action->tid;
                auto start_time = get_time(tid);
                if (last_time) {
                    auto o = event_time::is_ordered(last_time, start_time);
                    if (o != event_time::OrderBefore &&
                        (o != event_time::OrderEqual || last_is_start)) {
                        bb_throw_format(PyExc_ValueError, action_key(action->aid),
                                        "Actions on %U is not statically ordered",
                                        seq::channel_name_from_id(seqinfo, cid).get());
                    }
                }
                auto action_value = action->value.get();
                auto isramp = py_issubtype_nontrivial(Py_TYPE(action_value),
                                                      rampfunctionbase_type);
                last_is_start = !(action->is_pulse || isramp);
                last_time = last_is_start ? start_time : get_time(action->end_tid);
            }
        }
    }
    if (nchn > 0) {
        for (auto &val: cseq.basic_seqs[0]->start_values)
            val.reset(py_immref(Py_False));
        populate_bseq_values<_RampFunctionBase>(comp, cseq.basic_seqs[0]);
    }
    for (auto [name, backend]: pydict_iter<Backend>(comp->backends)) {
        throw_if(backend->__pyx_vtab->finalize(backend, cseq) < 0);
    }
}

static inline auto action_get_condval(auto action, unsigned age)
{
    auto cond = action->cond.get();
    if (cond == Py_True)
        return true;
    if (cond == Py_False)
        return false;
    assert(is_rtval(cond));
    try {
        rt_eval_throw((RuntimeValue*)cond, age);
        return !rtval_cache((RuntimeValue*)cond).is_zero();
    }
    catch (...) {
        bb_rethrow(action_key(action->aid));
    }
}

template<typename _RampFunctionBase, typename Backend>
static inline void compiler_runtime_finalize(auto comp, PyObject *_age,
                                             _RampFunctionBase*, Backend*)
{
    unsigned age = PyLong_AsLong(_age);
    throw_if(age == (unsigned)-1 && PyErr_Occurred());
    auto seq = comp->seq;
    auto &cseq = comp->cseq;
    using BasicSeq = std::remove_reference_t<decltype(*pyx_find_base(seq, bseq_id))>;
    auto seqinfo = pyx_fld(seq, seqinfo);
    auto bt_guard = set_global_tracker(&seqinfo->cinfo->bt_tracker);
    for (auto [i, bseq]: pylist_iter<BasicSeq>(pyx_fld(seq, basic_seqs))) {
        auto seqinfo = pyx_fld(bseq, seqinfo);
        auto time_mgr = seqinfo->time_mgr;
        assert(cseq.basic_seqs[i]->bseq_id == i);
        cseq.basic_seqs[i]->total_time =
            time_mgr->__pyx_vtab->compute_all_times(time_mgr, age);
    }
    for (auto [assert_id, a]: pylist_iter(seqinfo->assertions)) {
        auto c = (RuntimeValue*)PyTuple_GET_ITEM(a, 0);
        rt_eval_throw(c, age, assert_key(assert_id));
        if (rtval_cache(c).is_zero()) {
            bb_throw_format(PyExc_AssertionError, assert_key(assert_id),
                            "%U", PyTuple_GET_ITEM(a, 1));
        }
    }
    auto nchn = (int)PyList_GET_SIZE(seqinfo->channel_paths);
    for (auto *cbseq: cseq.basic_seqs) {
        auto bseq = (BasicSeq*)PyList_GET_ITEM(pyx_fld(seq, basic_seqs), cbseq->bseq_id);
        auto time_mgr = pyx_fld(bseq, seqinfo)->time_mgr;
        cbseq->total_time = cseq.basic_seqs[cbseq->bseq_id]->total_time;
        for (int cid = 0; cid < nchn; cid++) {
            auto &actions = cbseq->all_actions[cid];
            int64_t prev_time = 0;
            for (auto action: actions) {
                bool cond_val = action_get_condval(action, age);
                action->cond_val = cond_val;
                if (!cond_val)
                    continue;
                auto action_value = action->value.get();
                auto isramp = py_issubtype_nontrivial(Py_TYPE(action_value),
                                                      rampfunctionbase_type);
                if (isramp) {
                    auto rampf = (_RampFunctionBase*)action_value;
                    throw_if(rampf->__pyx_vtab->set_runtime_params(rampf, age),
                             action_key(action->aid));
                }
                else if (is_rtval(action_value)) {
                    rt_eval_throw((RuntimeValue*)action_value, age, action_key(action->aid));
                }
                auto action_end_val = action->end_val.get();
                if (action_end_val != action_value && is_rtval(action_end_val))
                    rt_eval_throw((RuntimeValue*)action_end_val, age,
                                  action_key(action->aid));
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
    for (auto [name, backend]: pydict_iter<Backend>(comp->backends)) {
        throw_if(backend->__pyx_vtab->runtime_finalize(backend, cseq, age) < 0);
    }
}

}
