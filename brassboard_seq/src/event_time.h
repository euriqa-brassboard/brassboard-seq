//

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
        assert(_is_static && !_is_rt_offset);
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
        assert(!_is_static && !_is_rt_offset);
        assume(!_is_static && !_is_rt_offset);
        return (long long)_c_offset;
    }

    inline void set_c_offset(long long value)
    {
        assert(!_is_static && !_is_rt_offset);
        assume(!_is_static && !_is_rt_offset);
        _c_offset = (uint64_t)value;
    }

    inline PyObject *get_rt_offset() const
    {
        if (_is_rt_offset)
            return assume(_rt_offset);
        return nullptr;
    }
    inline void set_rt_offset(PyObject *v)
    {
        assert(v && !_is_static && !_is_rt_offset);
        assume(v && !_is_static && !_is_rt_offset);
        _is_rt_offset = true;
        Py_INCREF(v);
        _rt_offset = v;
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

}

#endif
