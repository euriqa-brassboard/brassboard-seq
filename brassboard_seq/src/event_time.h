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

#ifndef BRASSBOARD_SEQ_SRC_EVENT_TIME_H
#define BRASSBOARD_SEQ_SRC_EVENT_TIME_H

#include "utils.h"

#include <assert.h>

namespace brassboard_seq::event_time {

using brassboard_seq::rtval::_RuntimeValue;

static constexpr long long time_scale = 1000000000000ll;

enum TimeOrder {
    NoOrder,
    OrderBefore,
    OrderEqual,
    OrderAfter,
};

struct EventTimeData {
    int id;
    bool floating: 1;
    bool _is_rt_offset: 1;
    int chain_id: 30;
private:
    union {
        struct {
            // Place this at the lowest bit so that when we are `_is_rt_offset`
            // this is never true due to the PyObject alignment.
            bool _is_static: 1;
            uint64_t _c_offset: 63;
        };
        PyObject *_rt_offset;
    };

public:
    EventTimeData()
        : _is_rt_offset(false),
          _is_static(false)
    {
    }
    EventTimeData(const EventTimeData&) = delete;
    EventTimeData(EventTimeData&&) = delete;
    EventTimeData &operator=(const EventTimeData&) = delete;
    EventTimeData &operator=(EventTimeData&&) = delete;
    ~EventTimeData()
    {
        if (_is_rt_offset) {
            Py_DECREF(_rt_offset);
        }
    }

    inline bool is_static() const
    {
        assert(!_is_static || !_is_rt_offset);
        return _is_static;
    }

    inline long long _get_static() const
    {
        assume(_is_static && !_is_rt_offset);
        return (long long)_c_offset;
    }
    inline long long get_static() const
    {
        if (_is_static) {
            assert(!_is_rt_offset);
            assume((long long)_c_offset != -1);
            return (long long)_c_offset;
        }
        return -1;
    }

    inline void set_static(long long value)
    {
        clear_rt_offset();
        _is_static = true;
        _c_offset = (uint64_t)value;
    }

    inline long long get_c_offset() const
    {
        assume(!_is_static && !_is_rt_offset);
        return (long long)_c_offset;
    }

    inline void set_c_offset(long long value)
    {
        assume(!_is_static && !_is_rt_offset);
        _c_offset = (uint64_t)value;
    }

    inline PyObject *get_rt_offset() const
    {
        if (_is_rt_offset)
            return assume(_rt_offset);
        return nullptr;
    }
    inline void set_rt_offset(auto *rv)
    {
        assume(rv && !_is_static && !_is_rt_offset);
        _is_rt_offset = true;
        assert(rv->datatype == rtval::DataType::Int64);
        _rt_offset = py_newref((PyObject*)rv);
        // Assume PyObject alignment
        assert(!_is_static);
    }
    inline void clear_rt_offset()
    {
        if (_is_rt_offset) {
            Py_DECREF(_rt_offset);
            _is_rt_offset = false;
        }
    }
};

static_assert(sizeof(EventTimeData) == 16);

struct TimeManagerStatus {
    int ntimes;
    bool finalized;
};

template<typename EventTime>
static inline __attribute__((returns_nonnull)) EventTime*
_new_time_int(auto *self, PyObject *EventTimeType, EventTime *prev,
              long long offset, bool floating, PyObject *cond, EventTime *wait_for)
{
    auto status = self->status.get();
    if (status->finalized)
        py_throw_format(PyExc_RuntimeError,
                        "Cannot allocate more time: already finalized");
    if (offset < 0)
        py_throw_format(PyExc_ValueError, "Time delay cannot be negative");
    py_object o(pytype_genericalloc(EventTimeType));
    auto tp = (EventTime*)o.get();
    new (&tp->manager_status) std::shared_ptr<TimeManagerStatus>(self->status);
    auto ntimes = status->ntimes;
    new (&tp->data) EventTimeData();
    tp->data.set_c_offset(offset);
    tp->data.floating = floating;
    tp->data.id = ntimes;
    new (&tp->chain_pos) std::vector<int> ();
    tp->prev = py_newref(prev);
    tp->wait_for = py_newref(wait_for);
    tp->cond = py_newref(cond);
    pylist_append(self->event_times, o.get());
    status->ntimes = ntimes + 1;
    o.release();
    return tp;
}

template<typename EventTime>
static inline __attribute__((returns_nonnull)) EventTime*
_new_time_rt(auto *self, PyObject *EventTimeType, EventTime *prev,
             _RuntimeValue *offset, PyObject *cond, EventTime *wait_for)
{
    auto status = self->status.get();
    if (status->finalized)
        py_throw_format(PyExc_RuntimeError,
                        "Cannot allocate more time: already finalized");
    py_object o(pytype_genericalloc(EventTimeType));
    auto tp = (EventTime*)o.get();
    new (&tp->manager_status) std::shared_ptr<TimeManagerStatus>(self->status);
    auto ntimes = status->ntimes;
    new (&tp->data) EventTimeData();
    tp->data.set_rt_offset(offset);
    tp->data.floating = false;
    tp->data.id = ntimes;
    new (&tp->chain_pos) std::vector<int> ();
    tp->prev = py_newref(prev);
    tp->wait_for = py_newref(wait_for);
    tp->cond = py_newref(cond);
    pylist_append(self->event_times, o.get());
    status->ntimes = ntimes + 1;
    o.release();
    return tp;
}

static inline long long round_time_f64(double v)
{
    return (long long)(v * double(time_scale) + 0.5);
}

static inline long long round_time_int(PyObject *v)
{
    if (Py_TYPE(v) == &PyLong_Type) {
        auto vi = PyLong_AsLongLong(v);
        throw_if(vi == -1 && PyErr_Occurred());
        return vi * time_scale;
    }
    return round_time_f64(get_value_f64(v, -1));
}

static inline __attribute__((returns_nonnull)) _RuntimeValue*
round_time_rt(_RuntimeValue *v, _RuntimeValue *rt_time_scale)
{
    py_object scaled_t((PyObject*)rtval::new_expr2(rtval::Mul, v, rt_time_scale));
    return rtval::rt_round_int64((_RuntimeValue*)scaled_t.get());
}

static inline TimeOrder is_ordered(auto *t1, auto *t2)
{
    auto manager_status = t1->manager_status.get();
    assert(manager_status == t2->manager_status.get());
    if (!manager_status->finalized)
        py_throw_format(PyExc_RuntimeError, "Event times not finalized");
    if (t1 == t2)
        return OrderEqual;
    auto chain1 = t1->data.chain_id;
    auto chain2 = t2->data.chain_id;
    // Assume t1 and t2 are on different chain if idx1 == idx2
    // Since otherwise they should've been the same time
    if (t2->chain_pos[chain1] >= t1->chain_pos[chain1])
        return OrderBefore;
    if (chain1 == chain2)
        return OrderAfter;
    if (t1->chain_pos[chain2] >= t2->chain_pos[chain2])
        return OrderAfter;
    return NoOrder;
}

}

#endif
