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

#include "rtval.h"

#include <assert.h>

namespace brassboard_seq::event_time {

using brassboard_seq::rtval::RuntimeValue;

// 1ps for internal time unit
static constexpr int64_t time_scale = 1000000000000ll;

static inline int64_t round_time_f64(double v)
{
    return round<int64_t>(v * double(time_scale));
}

static inline int64_t round_time_int(py::ptr<> v)
{
    if (v.typeis<py::int_>())
        return v.as_int<int64_t>() * time_scale;
    return round_time_f64(v.as_float());
}

rtval::rtval_ref round_time_rt(rtval::rtval_ptr v);

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
    // ID of the chain this time is part of
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
            py::DECREF(_rt_offset);
        }
    }

    inline bool is_static() const
    {
        assert(!_is_static || !_is_rt_offset);
        return _is_static;
    }

    inline int64_t _get_static() const
    {
        assume(_is_static && !_is_rt_offset);
        return (int64_t)_c_offset;
    }
    inline int64_t get_static() const
    {
        if (_is_static) {
            assert(!_is_rt_offset);
            assume((int64_t)_c_offset != -1);
            return (int64_t)_c_offset;
        }
        return -1;
    }

    inline void set_static(int64_t value)
    {
        clear_rt_offset();
        _is_static = true;
        _c_offset = (uint64_t)value;
    }

    inline int64_t get_c_offset() const
    {
        assume(!_is_static && !_is_rt_offset);
        return (int64_t)_c_offset;
    }

    inline void set_c_offset(int64_t value)
    {
        assume(!_is_static && !_is_rt_offset);
        _c_offset = (uint64_t)value;
    }

    inline rtval::rtval_ptr get_rt_offset() const
    {
        if (_is_rt_offset)
            return assume(_rt_offset);
        return rtval::rtval_ptr();
    }
    operator PyObject*() const
    {
        return (PyObject*)get_rt_offset();
    }
    template<typename T>
    inline void set_rt_offset(T &&rv)
    {
        assume(rv && !_is_static && !_is_rt_offset);
        _is_rt_offset = true;
        assert(rv->datatype == rtval::DataType::Int64);
        _rt_offset = py::newref(std::forward<T>(rv));
        // Assume PyObject alignment
        assert(!_is_static);
    }
    inline void clear_rt_offset()
    {
        if (_is_rt_offset) {
            py::DECREF(_rt_offset);
            _is_rt_offset = false;
        }
    }
};

static inline void CLEAR(EventTimeData &r)
{
    r.clear_rt_offset();
}

static_assert(sizeof(EventTimeData) == 16);

struct TimeManagerStatus {
    int ntimes;
    bool finalized;
};

struct EventTime;

using time_ptr = py::ptr<EventTime>;
using time_ref = py::ref<EventTime>;

struct TimeManager : PyObject {
    std::shared_ptr<TimeManagerStatus> status;
    py::list_ref event_times;
    std::vector<int64_t> time_values;
    // status:
    //   0: unevaluated
    //   1: evaluated but status/unchanged
    //   2: evaluated and changed
    std::vector<int8_t> time_status;

    void finalize();
    int64_t compute_all_times(unsigned age);

    time_ref new_int(time_ptr prev, int64_t offset, bool floating,
                     py::ptr<> cond, time_ptr wait_for);
    time_ref new_rt(time_ptr prev, rtval::rtval_ptr offset,
                    py::ptr<> cond, time_ptr wait_for);
    time_ref new_round(time_ptr prev, py::ptr<> offset, py::ptr<> cond, time_ptr wait_for);

    static PyTypeObject Type;
    static py::ref<TimeManager> alloc();
private:
    void visit_time(EventTime *t, auto &visited);
};

struct EventTime : PyObject {
    std::shared_ptr<TimeManagerStatus> manager_status;
    time_ref prev;
    time_ref wait_for;
    // If cond is false, this time point is the same as prev
    py::ref<> cond;
    EventTimeData data;
    // The largest index in each chain that we are no earlier than,
    // In particular for our own chain, this is the position we are in.
    std::vector<int> chain_pos;

    // All values are in units of `1/time_scale` seconds
    void set_base_int(time_ptr base, int64_t offset)
    {
        if (!data.floating)
            py_throw_format(PyExc_ValueError, "Cannot modify non-floating time");
        if (offset < 0)
            py_throw_format(PyExc_ValueError, "Time delay cannot be negative");
        prev.assign(base);
        data.set_c_offset(offset);
        data.floating = false;
    }
    template<typename T>
    void set_base_rt(time_ptr base, T &&offset)
    {
        if (!data.floating)
            py_throw_format(PyExc_ValueError, "Cannot modify non-floating time");
        prev.assign(base);
        data.set_rt_offset(std::forward<T>(offset));
        data.floating = false;
    }

    static PyTypeObject Type;
    int64_t get_value(int base_id, unsigned age, std::vector<int64_t> &cache,
                      std::vector<int8_t> &status);
private:
    void update_chain_pos(EventTime *prev, int nchains);
    friend struct TimeManager;
};

inline time_ref TimeManager::new_int(time_ptr prev, int64_t offset, bool floating,
                                     py::ptr<> cond, time_ptr wait_for)
{
    if (status->finalized)
        py_throw_format(PyExc_RuntimeError,
                        "Cannot allocate more time: already finalized");
    if (offset < 0)
        py_throw_format(PyExc_ValueError, "Time delay cannot be negative");
    auto tp = py::generic_alloc<EventTime>();
    call_constructor(&tp->manager_status, status);
    auto ntimes = status->ntimes;
    call_constructor(&tp->data);
    tp->data.set_c_offset(offset);
    tp->data.floating = floating;
    tp->data.id = ntimes;
    call_constructor(&tp->chain_pos);
    call_constructor(&tp->prev, py::newref(prev));
    call_constructor(&tp->wait_for, py::newref(wait_for));
    call_constructor(&tp->cond, py::newref(cond));
    event_times.append(tp);
    status->ntimes = ntimes + 1;
    return tp;
}

inline time_ref TimeManager::new_round(time_ptr prev, py::ptr<> offset,
                                       py::ptr<> cond, time_ptr wait_for)
{
    if (rtval::is_rtval(offset)) {
        return new_rt(prev, round_time_rt(offset), cond, wait_for);
    }
    else {
        auto coffset = round_time_int(offset);
        return new_int(prev, coffset, false, cond, wait_for);
    }
}

static inline TimeOrder is_ordered(EventTime *t1, EventTime *t2)
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

extern PyMethodDef methods[];

void init();

}

#endif
