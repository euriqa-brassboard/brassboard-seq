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

static PyTypeObject *timestep_type;
static PyTypeObject *rampfunctionbase_type;

using namespace rtval;

template<typename TimeStep, typename SubSeq>
static void collect_actions(SubSeq *self, std::vector<action::Action*> *actions)
{
    for (auto [i, subseq]: pylist_iter(self->sub_seqs)) {
        if (Py_TYPE(subseq) != timestep_type) {
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

template<typename TimeStep, typename _RampFunctionBase, typename Backend>
static inline void compiler_finalize(auto comp, TimeStep*, _RampFunctionBase*, Backend*)
{
    auto seq = comp->seq;
    using EventTime = std::remove_reference_t<decltype(*pyx_fld(seq, end_time))>;
    auto seqinfo = pyx_fld(seq, seqinfo);
    auto bt_guard = set_global_tracker(&seqinfo->bt_tracker);
    auto nchn = (int)PyList_GET_SIZE(seqinfo->channel_paths);
    for (auto [i, path]: pylist_iter(seqinfo->channel_paths)) {
        auto prefix = PyTuple_GET_ITEM(path, 0);
        auto res = PyDict_Contains(comp->backends, prefix);
        if (res > 0) [[likely]]
            continue;
        throw_if(res < 0);
        auto name = channel_name_from_path(path);
        py_throw_format(PyExc_ValueError, "Unhandled channel: %U", name.get());
    }
    auto time_mgr = seqinfo->time_mgr;
    time_mgr->__pyx_vtab->finalize(time_mgr);
    pyassign(seqinfo->channel_name_map, Py_None); // Free up memory
    auto all_actions = new std::vector<action::Action*>[nchn];
    comp->cseq.all_actions.reset(all_actions);
    collect_actions<TimeStep>(pyx_find_base(seq, sub_seqs), all_actions);
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
                auto name = seq::channel_name_from_id(seqinfo, cid);
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
                    auto name = seq::channel_name_from_id(seqinfo, cid);
                    bb_throw_format(PyExc_ValueError, action_key(action->aid),
                                    "Actions on %U is not statically ordered",
                                    name.get());
                }
            }
            auto action_value = action->value.get();
            auto isramp = py_issubtype_nontrivial(Py_TYPE(action_value),
                                                  rampfunctionbase_type);
            auto cond = action->cond.get();
            last_is_start = false;
            if (!action->is_pulse) {
                last_is_start = !isramp;
                if (cond != Py_False) {
                    py_object new_value;
                    if (isramp) {
                        auto rampf = (_RampFunctionBase*)action_value;
                        auto length = action->length;
                        auto vt = rampf->__pyx_vtab;
                        new_value.reset(throw_if_not(vt->eval_end(rampf, length, value),
                                                     action_key(action->aid)));
                    }
                    else {
                        new_value.reset(py_newref(action_value));
                    }
                    if (cond == Py_True) {
                        std::swap(value, new_value);
                    }
                    else if (new_value.get() != value.get()) {
                        assert(is_rtval(cond));
                        auto endval = _new_select((_RuntimeValue*)cond,
                                                  new_value, value);
                        value.reset((PyObject*)endval);
                    }
                }
            }
            last_time = last_is_start ? start_time : get_time(action->end_tid);
            action->end_val.reset(py_newref(value.get()));
        }
    }
    for (auto [name, _backend]: pydict_iter(comp->backends)) {
        auto backend = (Backend*)_backend;
        throw_if(backend->__pyx_vtab->finalize(backend, comp->cseq) < 0);
    }
}

static inline auto action_get_condval(auto action, unsigned age, py_object &pyage)
{
    auto cond = action->cond.get();
    if (cond == Py_True)
        return true;
    if (cond == Py_False)
        return false;
    assert(is_rtval(cond));
    try {
        rt_eval_throw((_RuntimeValue*)cond, age, pyage);
        return !rtval_cache((_RuntimeValue*)cond).is_zero();
    }
    catch (...) {
        bb_rethrow(action_key(action->aid));
    }
}

template<typename _RampFunctionBase, typename Backend>
static inline void compiler_runtime_finalize(auto comp, PyObject *_age,
                                             _RampFunctionBase*, Backend*)
{
    py_object pyage;
    if (Py_TYPE(_age) == &PyLong_Type)
        pyage.set_obj(_age);
    unsigned age = PyLong_AsLong(_age);
    throw_if(age == (unsigned)-1 && PyErr_Occurred());
    auto seq = comp->seq;
    auto seqinfo = pyx_fld(seq, seqinfo);
    auto bt_guard = set_global_tracker(&seqinfo->bt_tracker);
    auto time_mgr = seqinfo->time_mgr;
    comp->cseq.total_time = time_mgr->__pyx_vtab->compute_all_times(time_mgr, age, pyage);
    for (auto [assert_id, a]: pylist_iter(seqinfo->assertions)) {
        auto c = (_RuntimeValue*)PyTuple_GET_ITEM(a, 0);
        rt_eval_throw(c, age, pyage, assert_key(assert_id));
        if (rtval_cache(c).is_zero()) {
            bb_throw_format(PyExc_AssertionError, assert_key(assert_id),
                            "%U", PyTuple_GET_ITEM(a, 1));
        }
    }
    auto nchn = (int)PyList_GET_SIZE(seqinfo->channel_paths);
    for (int cid = 0; cid < nchn; cid++) {
        auto &actions = comp->cseq.all_actions[cid];
        long long prev_time = 0;
        for (auto action: actions) {
            bool cond_val = action_get_condval(action, age, pyage);
            action->cond_val = cond_val;
            if (!cond_val)
                continue;
            auto action_value = action->value.get();
            auto isramp = py_issubtype_nontrivial(Py_TYPE(action_value),
                                                  rampfunctionbase_type);
            if (isramp) {
                auto rampf = (_RampFunctionBase*)action_value;
                throw_if(rampf->__pyx_vtab->set_runtime_params(rampf, age, pyage),
                         action_key(action->aid));
            }
            else if (is_rtval(action_value)) {
                rt_eval_throw((_RuntimeValue*)action_value, age, pyage,
                              action_key(action->aid));
            }
            auto action_end_val = action->end_val.get();
            if (action_end_val != action_value && is_rtval(action_end_val)) {
                rt_eval_throw((_RuntimeValue*)action_end_val, age, pyage,
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
    for (auto [name, _backend]: pydict_iter(comp->backends)) {
        auto backend = (Backend*)_backend;
        throw_if(backend->__pyx_vtab->runtime_finalize(backend, comp->cseq, age, pyage) < 0);
    }
}

}
