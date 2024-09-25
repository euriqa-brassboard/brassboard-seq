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

#ifndef BRASSBOARD_SEQ_SRC_EVENT_TIME_H
#define BRASSBOARD_SEQ_SRC_EVENT_TIME_H

#include "utils.h"

#include <assert.h>

namespace brassboard_seq::event_time {

static constexpr long long time_scale = 1000000000000ll;

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
    template<typename RuntimeValue>
    inline void set_rt_offset(RuntimeValue *v)
    {
        assume(v && !_is_static && !_is_rt_offset);
        _is_rt_offset = true;
        assert(v->datatype == rtval::DataType::Int64);
        _rt_offset = py_newref((PyObject*)v);
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

template<typename TimeManager, typename EventTime>
static inline __attribute__((returns_nonnull)) EventTime*
_new_time_int(TimeManager *self, PyObject *EventTimeType, EventTime *prev,
              long long offset, bool floating, PyObject *cond, EventTime *wait_for)
{
    auto status = self->status.get();
    if (status->finalized) {
        PyErr_Format(PyExc_RuntimeError,
                     "Cannot allocate more time: already finalized");
        throw 0;
    }
    if (offset < 0) {
        PyErr_Format(PyExc_ValueError, "Time delay cannot be negative");
        throw 0;
    }
    py_object o(throw_if_not(PyType_GenericAlloc((PyTypeObject*)EventTimeType, 0)));
    auto tp = (EventTime*)o.get();
    new (&tp->manager_status) std::shared_ptr<TimeManagerStatus>(self->status);
    auto ntimes = status->ntimes;
    new (&tp->data) EventTimeData();
    tp->data.set_c_offset(offset);
    tp->data.floating = floating;
    tp->data.id = ntimes;
    new (&tp->chain_pos) std::vector<int> ();
    tp->prev = (EventTime*)py_newref((PyObject*)prev);
    tp->wait_for = (EventTime*)py_newref((PyObject*)wait_for);
    tp->cond = py_newref(cond);
    throw_if(pylist_append(self->event_times, o.get()));
    status->ntimes = ntimes + 1;
    o.release();
    return tp;
}

template<typename TimeManager, typename EventTime, typename RuntimeValue>
static inline __attribute__((returns_nonnull)) EventTime*
_new_time_rt(TimeManager *self, PyObject *EventTimeType, EventTime *prev,
             RuntimeValue *offset, PyObject *cond, EventTime *wait_for)
{
    auto status = self->status.get();
    if (status->finalized) {
        PyErr_Format(PyExc_RuntimeError, "Cannot allocate more time: already finalized");
        throw 0;
    }
    py_object o(throw_if_not(PyType_GenericAlloc((PyTypeObject*)EventTimeType, 0)));
    auto tp = (EventTime*)o.get();
    new (&tp->manager_status) std::shared_ptr<TimeManagerStatus>(self->status);
    auto ntimes = status->ntimes;
    new (&tp->data) EventTimeData();
    tp->data.set_rt_offset(offset);
    tp->data.floating = false;
    tp->data.id = ntimes;
    new (&tp->chain_pos) std::vector<int> ();
    tp->prev = (EventTime*)py_newref((PyObject*)prev);
    tp->wait_for = (EventTime*)py_newref((PyObject*)wait_for);
    tp->cond = py_newref(cond);
    throw_if(pylist_append(self->event_times, o.get()));
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
    auto vf = PyFloat_AsDouble(v);
    throw_if(vf == -1 && PyErr_Occurred());
    return round_time_f64(vf);
}

template<typename RuntimeValue>
static inline __attribute__((returns_nonnull)) RuntimeValue*
round_time_rt(PyObject *RTValueType, RuntimeValue *v, RuntimeValue *rt_time_scale)
{
    py_object scaled_t((PyObject*)rtval::_new_expr2(RTValueType, rtval::Mul, v,
                                                    rt_time_scale));
    return rtval::rt_round_int64(RTValueType, (RuntimeValue*)scaled_t.get());
}

}

#endif
