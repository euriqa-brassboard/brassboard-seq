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

#include "event_time.h"

#include "Python.h"

#include "assert.h"

#include <array>
#include <unordered_set>

namespace brassboard_seq::event_time {

static inline bool get_cond_val(PyObject *v, unsigned age, py_object &pyage)
{
    if (v == Py_True) [[likely]]
        return true;
    if (v == Py_False)
        return false;
    assert(rtval::is_rtval(v));
    auto rv = (rtval::_RuntimeValue*)v;
    rtval::rt_eval_throw(rv, age, pyage);
    return !rtval::rtval_cache(rv).is_zero();
}

static PyObject *str_time(long long t)
{
    assert(time_scale == 1e12);
    assert(t >= 0);

    std::array<char, 32> str;
    auto ptr = to_chars(str, t);
    *ptr = 0;

    auto s = time_scale;
    auto ms = s / 1000;
    auto us = ms / 1000;
    auto ns = us / 1000;

    int dec = 0;
    const char *unit = "ps";
    if (t >= s / 10 * 3) {
        dec = 12;
        unit = "s";
    }
    else if (t >= ms / 10 * 3) {
        dec = 9;
        unit = "ms";
    }
    else if (t >= us / 10 * 3) {
        dec = 6;
        unit = "us";
    }
    else if (t >= ns / 10 * 3) {
        dec = 3;
        unit = "ns";
    }
    char *end = ptr;
    assert(end - str.data() >= dec);
    for (int i = 0; i < dec; i++) {
        if (ptr[-1 - i] != '0')
            break;
        end = &ptr[-1 - i];
        *end = 0;
    }
    auto pdec = ptr - dec;
    if (pdec == str.data()) {
        memmove(pdec + 2, pdec, end - pdec);
        pdec[0] = '0';
        pdec[1] = '.';
        end += 2;
    }
    else if (pdec != end) {
        memmove(pdec + 1, pdec, end - pdec);
        pdec[0] = '.';
        end += 1;
    }
    *end = ' ';
    memcpy(&end[1], unit, strlen(unit) + 1);
    return PyUnicode_FromString(str.data());
}

static inline void visit_time(auto self, auto t, auto &visited)
{
    auto id = t->data.id;
    if (id < 0)
        return;
    if (t->data.floating)
        bb_throw_format(PyExc_RuntimeError, event_time_key(t),
                        "Event time still floating");
    if (visited.contains(id))
        bb_throw_format(PyExc_ValueError, event_time_key(t), "Time loop detected");
    visited.insert(id);
    long long static_offset;
    auto rt_offset = t->data.get_rt_offset();
    auto cond = t->cond;
    if (cond == Py_True) {
        static_offset = rt_offset ? -1 : t->data.get_c_offset();
    }
    else if (cond == Py_False || (!rt_offset && t->data.get_c_offset() == 0)) {
        static_offset = 0;
    }
    else {
        static_offset = -1;
    }
    long long static_prev = 0;
    long long static_wait_for = 0;
    if (auto prev = t->prev; (PyObject*)prev != Py_None) {
        visit_time(self, prev, visited);
        static_prev = prev->data.get_static();
    }
    if (auto wait_for = t->wait_for; (PyObject*)wait_for != Py_None) {
        visit_time(self, wait_for, visited);
        static_wait_for = wait_for->data.get_static();
        if (cond == Py_False) {
            if (static_prev != -1) {
                t->data.set_static(static_prev);
            }
        }
        else if (cond == Py_True && static_wait_for != -1 &&
                 static_offset != -1 && static_prev != -1) {
            t->data.set_static(std::max(static_prev, static_wait_for + static_offset));
        }
    }
    else if (static_offset != -1 && static_prev != -1) {
        t->data.set_static(static_prev + static_offset);
    }
    t->data.id = -1;
    pylist_append(self->event_times, (PyObject*)t);
}

static inline void update_chain_pos(auto self, auto prev, int nchains)
{
    auto cid = self->data.chain_id;
    for (int i = 0; i < nchains; i++) {
        if (i == cid)
            continue;
        self->chain_pos[i] = std::max(self->chain_pos[i], prev->chain_pos[i]);
    }
}

// If the base time has a static value, the returned time values will be the actual
// time point of the EventTime. If the base time does not have a static value
// the time offset relative to the base time is returned.
// This is so that if the base time is not statically known,
// we can compute the diff without computing the base time,
// while if the base time is known, we can use the static values in the computation
static inline long long get_time_value(auto self, int base_id, unsigned age,
                                       py_object &pyage, std::vector<long long> &cache)
{
    auto tid = self->data.id;
    if (tid == base_id)
        return self->data.is_static() ? self->data._get_static() : 0;
    assert(tid > base_id);
    auto value = cache[tid];
    if (value >= 0)
        return value;
    // The time manager should've been finalized and
    // no time should be floating anymore
    assert(!self->data.floating);

    if (self->data.is_static()) {
        // If we have a static value it means that the base time has a static value
        // In this case, we are returning the full time and there's no need to
        // compute the offset from the base time.
        auto static_value = self->data._get_static();
        cache[tid] = static_value;
        return static_value;
    }

    long long prev_val = 0;
    if (auto prev = self->prev; (PyObject*)prev != Py_None)
        prev_val = get_time_value(prev, base_id, age, pyage, cache);

    auto cond = get_cond_val(self->cond, age, pyage);
    long long offset = 0;
    if (cond) {
        auto rt_offset = (rtval::_RuntimeValue*)self->data.get_rt_offset();
        if (rt_offset) {
            rt_eval_throw(rt_offset, age, pyage, event_time_key(self));
            offset = rt_offset->cache_val.i64_val;
            if (offset < 0) {
                bb_throw_format(PyExc_ValueError, event_time_key(self),
                                "Time delay cannot be negative");
            }
        }
        else {
            offset = self->data.get_c_offset();
        }
    }

    if (auto wait_for = self->wait_for; (PyObject*)wait_for == Py_None) {
        // When wait_for is None, the offset is added to the previous time
        value = prev_val + offset;
    }
    else {
        // Otherwise, the wait_for is added to the wait_for time.
        value = prev_val;
        // Do not try to evaluate wait_for unless the condition is true
        // When a base_id is supplied, the wait_for event time may not share
        // this base if the condition isn't true.
        if (cond) {
            value = std::max(value, get_time_value(wait_for, base_id, age,
                                                   pyage, cache) + offset);
        }
    }

    cache[tid] = value;
    return value;
}

template<typename EventTime>
static inline void timemanager_finalize(auto self, EventTime*)
{
    if (self->status->finalized)
        py_throw_format(PyExc_RuntimeError, "Event times already finalized");
    self->status->finalized = true;
    auto event_times = pylist_new(0);
    py_object old_event_times(self->event_times); // steal reference
    self->event_times = event_times;
    std::unordered_set<int> visited;
    // First, topologically order the times
    for (auto [i, t]: pylist_iter(old_event_times.get()))
        visit_time(self, (EventTime*)t, visited);

    std::vector<int> chain_lengths;
    for (auto [tid, t]: pylist_iter<EventTime>(event_times)) {
        t->data.id = tid;
        int chain_id1 = -1;
        int chain_pos1;
        if (auto prev = t->prev; (PyObject*)prev != Py_None) {
            chain_id1 = prev->data.chain_id;
            chain_pos1 = prev->chain_pos[chain_id1];
            if (chain_pos1 + 1 != chain_lengths[chain_id1]) {
                // Not the tail of the chain
                chain_id1 = -1;
            }
        }
        int chain_id2 = -1;
        int chain_pos2;
        if (auto wait_for = t->wait_for; (PyObject*)wait_for != Py_None) {
            chain_id2 = wait_for->data.chain_id;
            chain_pos2 = wait_for->chain_pos[chain_id2];
            if (chain_pos2 + 1 != chain_lengths[chain_id2]) {
                // Not the tail of the chain
                chain_id2 = -1;
            }
        }
        int chain_id;
        int chain_pos;
        if (chain_id1 == -1) {
            if (chain_id2 == -1) {
                // New chain
                chain_id = chain_lengths.size();
                chain_pos = 0;
                chain_lengths.push_back(0);
            }
            else {
                chain_id = chain_id2;
                chain_pos = chain_pos2 + 1;
            }
        }
        else if (chain_id2 == -1) {
            chain_id = chain_id1;
            chain_pos = chain_pos1 + 1;
        }
        else if (chain_pos2 > chain_pos1) {
            chain_id = chain_id2;
            chain_pos = chain_pos2 + 1;
        }
        else {
            chain_id = chain_id1;
            chain_pos = chain_pos1 + 1;
        }
        chain_lengths[chain_id] = chain_pos + 1;
        t->data.chain_id = chain_id;
        t->chain_pos.resize(chain_lengths.size(), -1);
        t->chain_pos[chain_id] = chain_pos;
    }

    int nchains = chain_lengths.size();
    for (auto [tid, t]: pylist_iter<EventTime>(event_times)) {
        t->chain_pos.resize(nchains, -1);
        if (auto prev = t->prev; (PyObject*)prev != Py_None)
            update_chain_pos(t, prev, nchains);
        if (auto wait_for = t->wait_for; (PyObject*)wait_for != Py_None)
            update_chain_pos(t, wait_for, nchains);
    }
}

template<typename EventTime>
static inline long long timemanager_compute_all_times(auto self, unsigned age,
                                                      py_object &pyage, EventTime*)
{
    if (!self->status->finalized)
        py_throw_format(PyExc_RuntimeError, "Event times not finalized");
    long long max_time = 0;
    auto event_times = self->event_times;
    int ntimes = PyList_GET_SIZE(event_times);
    self->time_values.resize(ntimes);
    std::ranges::fill(self->time_values, -1);
    for (auto [i, t]: pylist_iter<EventTime>(event_times))
        max_time = std::max(max_time, get_time_value(t, -1, age, pyage,
                                                     self->time_values));
    return max_time;
}

// Returns borrowed reference
template<typename EventTime>
static inline int eventtime_find_base_id(EventTime *t1, EventTime *t2,
                                         unsigned age, py_object &pyage)
{
    std::map<int,EventTime*> frontier;
    if (!t1->manager_status->finalized)
        py_throw_format(PyExc_RuntimeError, "Event times not finalized");
    frontier[t1->data.id] = t1;
    frontier[t2->data.id] = t2;
    while (frontier.size() > 1) {
        auto it = --frontier.end();
        auto t = it->second;
        frontier.erase(it);
        auto prev = t->prev;
        if ((PyObject*)prev == Py_None)
            return -1; // Found the start of the experiment
        frontier[prev->data.id] = prev;
        if (auto wait_for = t->wait_for;
            (PyObject*)wait_for != Py_None && get_cond_val(t->cond, age, pyage)) {
            frontier[wait_for->data.id] = wait_for;
        }
    }
    return frontier.begin()->second->data.id;
}

static inline double timediff_eval(auto self, unsigned age)
{
    auto t1 = self->t1;
    auto t2 = self->t2;
    if (self->in_eval)
        py_throw_format(PyExc_ValueError, "Recursive value dependency detected.");
    self->in_eval = true;
    ScopeExit reset_eval([&] { self->in_eval = false; });
    py_object pyage;
    int base_id = eventtime_find_base_id(t1, t2, age, pyage);
    std::vector<long long> cache(t1->manager_status->ntimes, -1);
    return double(get_time_value(t1, base_id, age, pyage, cache) -
                  get_time_value(t2, base_id, age, pyage, cache)) / time_scale;
}

static traverseproc event_time_base_traverse;
static inquiry event_time_base_clear;

template<typename EventTime>
static inline void update_event_time_gc_callback(PyObject *_type, EventTime*)
{
    auto type = (PyTypeObject*)_type;
    event_time_base_traverse = type->tp_traverse;
    event_time_base_clear = type->tp_clear;
    type->tp_traverse = [] (PyObject *obj, visitproc visit, void *arg) -> int {
        auto t = (EventTime*)obj;
        Py_VISIT(t->data.get_rt_offset());
        return event_time_base_traverse(obj, visit, arg);
    };
    type->tp_clear = [] (PyObject *obj) -> int {
        auto t = (EventTime*)obj;
        t->data.clear_rt_offset();
        return event_time_base_clear(obj);
    };
}

}
