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

#include "assert.h"

#include <array>
#include <unordered_set>

namespace brassboard_seq::event_time {

__attribute__((visibility("protected"),returns_nonnull))
PyObject *py_time_scale()
{
    static auto val = py::new_int(time_scale).rel();
    return val;
}

__attribute__((visibility("protected"),returns_nonnull))
RuntimeValue *round_time_rt(RuntimeValue *v)
{
    static RuntimeValue *rt_scale = rtval::new_const(py_time_scale());
    return rtval::rt_round_int64(py::ref(rtval::new_expr2(rtval::Mul, v, rt_scale)).get());
}

static inline bool get_cond_val(PyObject *v, unsigned age)
{
    if (v == Py_True) [[likely]]
        return true;
    if (v == Py_False)
        return false;
    assert(rtval::is_rtval(v));
    auto rv = (RuntimeValue*)v;
    rtval::rt_eval_throw(rv, age);
    return !rtval::rtval_cache(rv).is_zero();
}

static py::str_ref str_time(int64_t t)
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
    return py::new_str(str.data());
}

inline void TimeManager::visit_time(EventTime *t, auto &visited)
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
    int64_t static_offset;
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
    int64_t static_prev = 0;
    int64_t static_wait_for = 0;
    if (auto prev = t->prev; prev != Py_None) {
        visit_time(prev, visited);
        static_prev = prev->data.get_static();
    }
    if (auto wait_for = t->wait_for; wait_for != Py_None) {
        visit_time(wait_for, visited);
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
    py::list(event_times).append(t);
}

__attribute__((visibility("protected"),returns_nonnull))
EventTime *TimeManager::new_rt(EventTime *prev, RuntimeValue *offset,
                               PyObject *cond, EventTime *wait_for)
{
    if (status->finalized)
        py_throw_format(PyExc_RuntimeError,
                        "Cannot allocate more time: already finalized");
    auto tp = py::generic_alloc<EventTime>();
    call_constructor(&tp->manager_status, status);
    auto ntimes = status->ntimes;
    call_constructor(&tp->data);
    tp->data.set_rt_offset(offset);
    tp->data.floating = false;
    tp->data.id = ntimes;
    call_constructor(&tp->chain_pos);
    tp->prev = py::newref(prev);
    tp->wait_for = py::newref(wait_for);
    tp->cond = py::newref(cond);
    py::list(event_times).append(tp);
    status->ntimes = ntimes + 1;
    return tp.rel();
}

__attribute__((visibility("protected")))
void TimeManager::finalize()
{
    if (status->finalized)
        py_throw_format(PyExc_RuntimeError, "Event times already finalized");
    status->finalized = true;
    auto _event_times = py::new_list(0);
    py::ref old_event_times(event_times); // steal reference
    event_times = _event_times.rel();

    std::unordered_set<int> visited;
    // First, topologically order the times
    for (auto [i, t]: py::list_iter<EventTime>(old_event_times))
        visit_time(t, visited);

    std::vector<int> chain_lengths;
    for (auto [tid, t]: py::list_iter<EventTime>(event_times)) {
        t->data.id = tid;
        int chain_id1 = -1;
        int chain_pos1;
        if (auto prev = t->prev; prev != Py_None) {
            chain_id1 = prev->data.chain_id;
            chain_pos1 = prev->chain_pos[chain_id1];
            if (chain_pos1 + 1 != chain_lengths[chain_id1]) {
                // Not the tail of the chain
                chain_id1 = -1;
            }
        }
        int chain_id2 = -1;
        int chain_pos2;
        if (auto wait_for = t->wait_for; wait_for != Py_None) {
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
    for (auto [tid, t]: py::list_iter<EventTime>(event_times)) {
        t->chain_pos.resize(nchains, -1);
        if (auto prev = t->prev; prev != Py_None)
            t->update_chain_pos(prev, nchains);
        if (auto wait_for = t->wait_for; wait_for != Py_None)
            t->update_chain_pos(wait_for, nchains);
    }
}

__attribute__((visibility("protected")))
int64_t TimeManager::compute_all_times(unsigned age)
{
    if (!status->finalized)
        py_throw_format(PyExc_RuntimeError, "Event times not finalized");
    int64_t max_time = 0;
    int ntimes = py::list(event_times).size();
    time_values.resize(ntimes);
    std::ranges::fill(time_values, -1);
    for (auto [i, t]: py::list_iter<EventTime>(event_times))
        max_time = std::max(max_time, t->get_value(-1, age, time_values));
    return max_time;
}

__attribute__((returns_nonnull,visibility("protected")))
TimeManager *TimeManager::alloc()
{
    auto self = py::generic_alloc<TimeManager>();
    self->event_times = py::new_list(0).rel();
    auto status = new TimeManagerStatus;
    status->finalized = false;
    status->ntimes = 0;
    call_constructor(&self->status, status);
    call_constructor(&self->time_values);
    return self.rel();
}

__attribute__((visibility("protected")))
PyTypeObject TimeManager::Type = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "brassboard_seq.event_time.TimeManager",
    .tp_basicsize = sizeof(TimeManager),
    .tp_dealloc = [] (PyObject *py_self) {
        auto self = (TimeManager*)py_self;
        PyObject_GC_UnTrack(self);
        Type.tp_clear(self);
        call_destructor(&self->status);
        call_destructor(&self->time_values);
        Py_TYPE(self)->tp_free(self);
    },
    .tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_HAVE_GC,
    .tp_traverse = [] (PyObject *py_self, visitproc visit, void *arg) {
        auto self = (TimeManager*)py_self;
        Py_VISIT(self->event_times);
        return 0;
    },
    .tp_clear = [] (PyObject *py_self) {
        auto self = (TimeManager*)py_self;
        Py_CLEAR(self->event_times);
        return 0;
    },
};

namespace {

static PyObject *eventtime_str(PyObject *py_self)
{
    return cxx_catch([&] {
        auto self = (EventTime*)py_self;
        if (self->data.floating)
            return "<floating>"_py.ref();
        if (self->data.is_static())
            return str_time(self->data._get_static());
        auto rt_offset = py::ptr(self->data.get_rt_offset());
        auto str_offset(rt_offset ? rt_offset.str() : str_time(self->data.get_c_offset()));
        auto prev = self->prev;
        auto cond = self->cond;
        if (prev == Py_None) {
            assert(cond == Py_True);
            return str_offset;
        }
        auto wait_for = self->wait_for;
        if (wait_for == Py_None) {
            if (cond == Py_True)
                return py::str_format("T[%u] + %U", prev->data.id, str_offset);
            return py::str_format("T[%u] + (%U; if %S)", prev->data.id, str_offset, cond);
        }
        if (cond == Py_True)
            return py::str_format("T[%u]; wait_for(T[%u] + %U)",
                                  prev->data.id, wait_for->data.id, str_offset);
        return py::str_format("T[%u]; wait_for(T[%u] + %U; if %S)",
                              prev->data.id, wait_for->data.id, str_offset, cond);
    });
}

static inline int eventtime_find_base_id(EventTime *t1, EventTime *t2, unsigned age)
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
        if (prev == Py_None)
            return -1; // Found the start of the experiment
        frontier[prev->data.id] = prev;
        if (auto wait_for = t->wait_for;
            wait_for != Py_None && get_cond_val(t->cond, age)) {
            frontier[wait_for->data.id] = wait_for;
        }
    }
    return frontier.begin()->second->data.id;
}

struct EventTimeDiff : rtval::ExternCallback {
    EventTime *t1;
    EventTime *t2;
    bool in_eval;

    static rtval::TagVal eval(EventTimeDiff *self, unsigned age)
    {
        auto t1 = self->t1;
        auto t2 = self->t2;
        if (self->in_eval)
            py_throw_format(PyExc_ValueError, "Recursive value dependency detected.");
        self->in_eval = true;
        ScopeExit reset_eval([&] { self->in_eval = false; });
        int base_id = eventtime_find_base_id(t1, t2, age);
        std::vector<int64_t> cache(t1->manager_status->ntimes, -1);
        return double(t1->get_value(base_id, age, cache) -
                      t2->get_value(base_id, age, cache)) / time_scale;
    }

    static PyTypeObject Type;
};
PyTypeObject EventTimeDiff::Type = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "brassboard_seq.event_time.EventTimeDiff",
    .tp_basicsize = sizeof(EventTimeDiff),
    .tp_dealloc = [] (PyObject *py_self) {
        PyObject_GC_UnTrack(py_self);
        Type.tp_clear(py_self);
        Py_TYPE(py_self)->tp_free(py_self);
    },
    .tp_str = [] (PyObject *py_self) {
        auto self = (EventTimeDiff*)py_self;
        return PyUnicode_FromFormat("(T[%u] - T[%u])",
                                    self->t1->data.id, self->t2->data.id);
    },
    .tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_HAVE_GC,
    .tp_traverse = [] (PyObject *py_self, visitproc visit, void *arg) {
        auto self = (EventTimeDiff*)py_self;
        Py_VISIT(self->t1);
        Py_VISIT(self->t2);
        return 0;
    },
    .tp_clear = [] (PyObject *py_self) {
        auto self = (EventTimeDiff*)py_self;
        Py_CLEAR(self->t1);
        Py_CLEAR(self->t2);
        return 0;
    },
};

static PyNumberMethods EventTime_as_number = {
    .nb_subtract = [] (PyObject *v1, PyObject *v2) {
        auto self = (EventTime*)v1;
        auto other = (EventTime*)v2;
        return cxx_catch([&] {
            if (self->manager_status != other->manager_status)
                py_throw_format(PyExc_ValueError,
                                "Cannot take the difference between unrelated times");
            auto diff = py::generic_alloc<EventTimeDiff>();
            diff->t1 = (EventTime*)py::newref(self);
            diff->t2 = (EventTime*)py::newref(other);
            diff->in_eval = false;
            diff->fptr = (void*)EventTimeDiff::eval;
            return (PyObject*)rtval::new_extern_age(diff.get(),
                                                    (PyObject*)&PyFloat_Type);
        });
    },
};

}

// If the base time has a static value, the returned time values will be the actual
// time point of the EventTime. If the base time does not have a static value
// the time offset relative to the base time is returned.
// This is so that if the base time is not statically known,
// we can compute the diff without computing the base time,
// while if the base time is known, we can use the static values in the computation
inline int64_t EventTime::get_value(int base_id, unsigned age, std::vector<int64_t> &cache)
{
    auto tid = data.id;
    if (tid == base_id)
        return data.is_static() ? data._get_static() : 0;
    assert(tid > base_id);
    auto value = cache[tid];
    if (value >= 0)
        return value;
    // The time manager should've been finalized and
    // no time should be floating anymore
    assert(!data.floating);

    if (data.is_static()) {
        // If we have a static value it means that the base time has a static value
        // In this case, we are returning the full time and there's no need to
        // compute the offset from the base time.
        auto static_value = data._get_static();
        cache[tid] = static_value;
        return static_value;
    }

    int64_t prev_val = 0;
    if (prev != Py_None)
        prev_val = prev->get_value(base_id, age, cache);

    auto cond_val = get_cond_val(cond, age);
    int64_t offset = 0;
    if (cond_val) {
        auto rt_offset = (RuntimeValue*)data.get_rt_offset();
        if (rt_offset) {
            rtval::rt_eval_throw(rt_offset, age, event_time_key(this));
            offset = rt_offset->cache_val.i64_val;
            if (offset < 0) {
                bb_throw_format(PyExc_ValueError, event_time_key(this),
                                "Time delay cannot be negative");
            }
        }
        else {
            offset = data.get_c_offset();
        }
    }

    if (wait_for == Py_None) {
        // When wait_for is None, the offset is added to the previous time
        value = prev_val + offset;
    }
    else {
        // Otherwise, the wait_for is added to the wait_for time.
        value = prev_val;
        // Do not try to evaluate wait_for unless the condition is true
        // When a base_id is supplied, the wait_for event time may not share
        // this base if the condition isn't true.
        if (cond_val) {
            value = std::max(value, wait_for->get_value(base_id, age, cache) + offset);
        }
    }

    cache[tid] = value;
    return value;
}

inline void EventTime::update_chain_pos(EventTime *prev, int nchains)
{
    auto cid = data.chain_id;
    for (int i = 0; i < nchains; i++) {
        if (i == cid)
            continue;
        chain_pos[i] = std::max(chain_pos[i], prev->chain_pos[i]);
    }
}

__attribute__((visibility("protected")))
PyTypeObject EventTime::Type = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "brassboard_seq.event_time.EventTime",
    .tp_basicsize = sizeof(EventTime),
    .tp_dealloc = [] (PyObject *py_self) {
        auto self = (EventTime*)py_self;
        PyObject_GC_UnTrack(self);
        Type.tp_clear(self);
        call_destructor(&self->manager_status);
        call_destructor(&self->data);
        call_destructor(&self->chain_pos);
        Py_TYPE(self)->tp_free(self);
    },
    .tp_repr = eventtime_str,
    .tp_as_number = &EventTime_as_number,
    .tp_str = eventtime_str,
    .tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_HAVE_GC,
    .tp_traverse = [] (PyObject *obj, visitproc visit, void *arg) -> int {
        auto t = (EventTime*)obj;
        Py_VISIT(t->prev);
        Py_VISIT(t->wait_for);
        Py_VISIT(t->cond);
        Py_VISIT(t->data.get_rt_offset());
        return 0;
    },
    .tp_clear = [] (PyObject *obj) -> int {
        auto t = (EventTime*)obj;
        t->data.clear_rt_offset();
        Py_CLEAR(t->prev);
        Py_CLEAR(t->wait_for);
        Py_CLEAR(t->cond);
        return 0;
    },
};

__attribute__((constructor))
static void init()
{
    throw_if(PyType_Ready(&TimeManager::Type) < 0);
    throw_if(PyType_Ready(&EventTime::Type) < 0);
    throw_if(PyType_Ready(&EventTimeDiff::Type) < 0);
}

}
