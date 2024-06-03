# cython: language_level=3

# Do not use relative import since it messes up cython file name tracking
from brassboard_seq.rtval cimport ifelse, get_value, \
  new_const, new_extern_age, rt_eval
from brassboard_seq.utils cimport _assume_not_none, \
  event_time_key, bb_err_format

from libcpp.map cimport map as cppmap
from libcpp.algorithm cimport fill as cppfill

cimport cython
from cython.operator cimport dereference as deref, predecrement as predec

cdef object py_time_scale = c_time_scale
cdef RuntimeValue rt_time_scale = new_const(py_time_scale)

def time_scale():
    return py_time_scale

cdef extern from *:
    """
#include "Python.h"

#include "assert.h"

#include <array>
#include <charconv>

PyObject *_str_time(long long t)
{
    assert(c_time_scale == 1e12);
    assert(t >= 0);

    std::array<char, 32> str;

    auto [ptr, ec] = std::to_chars(str.data(), str.data() + str.size(), t);
    if (ec != std::errc()) {
        PyErr_Format(PyExc_RuntimeError, "%s",
                     std::make_error_code(ec).message().c_str());
        return nullptr;
    }
    *ptr = 0;

    auto s = c_time_scale;
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
    """
    cdef str _str_time(long long)

cdef int visit_time(TimeManager self, EventTime t, unordered_set[int] &visited) except -1:
    cdef int id = t.data.id
    if id < 0:
        return 0
    if t.data.floating:
        bb_err_format(RuntimeError, event_time_key(<void*>t),
                      "Event time still floating")
    if visited.count(id):
        bb_err_format(ValueError, event_time_key(<void*>t), "Time loop detected")
    visited.insert(id)
    cdef long long static_offset
    rt_offset = t.rt_offset
    cond = t.cond
    if cond is True:
        if rt_offset is not None:
            static_offset = -1
        else:
            static_offset = <long long>t.data.c_offset
    elif cond is False:
        static_offset = 0
    elif rt_offset is None and t.data.c_offset == 0:
        static_offset = 0
    else:
        static_offset = -1

    cdef long long static_prev = 0
    prev = t.prev
    if prev is not None:
        visit_time(self, prev, visited)
        static_prev = prev.data.get_static()
    wait_for = t.wait_for
    if wait_for is not None:
        visit_time(self, wait_for, visited)
        static_wait_for = wait_for.data.get_static()
        if cond is False:
            if static_prev != -1:
                t.data.set_static(static_prev)
        elif (cond is True and static_wait_for != -1 and
              static_offset != -1 and static_prev != -1):
            t.data.set_static(max(static_prev, static_wait_for + static_offset))
    elif static_offset != -1 and static_prev != -1:
        t.data.set_static(static_prev + static_offset)
    t.data.id = -1
    _assume_not_none(<void*>self.event_times)
    self.event_times.append(t)
    return 0

@cython.final
cdef class TimeManager:
    def __init__(self):
        PyErr_Format(TypeError, "TimeManager cannot be created directly")

    cdef int finalize(self) except -1:
        status = self.status.get()
        if status.finalized:
            PyErr_Format(RuntimeError, "Event times already finalized")
        status.finalized = True
        old_event_times = self.event_times
        event_times = []
        self.event_times = event_times
        cdef unordered_set[int] visited
        # First, topologically order the times
        assume_not_none(old_event_times)
        for _t in old_event_times:
            visit_time(self, <EventTime>_t, visited)

        cdef vector[int] chain_lengths
        cdef int i = 0
        cdef int chain_id1
        cdef int chain_pos1
        cdef int chain_id2
        cdef int chain_pos2
        cdef int chain_id
        cdef int chain_pos
        cdef EventTime t
        assume_not_none(event_times)
        for _t in event_times:
            t = <EventTime>_t
            tid = i
            i += 1
            t.data.id = tid
            prev = t.prev
            if prev is not None:
                chain_id1 = prev.data.chain_id
                chain_pos1 = prev.chain_pos[chain_id1]
                if chain_pos1 + 1 != chain_lengths[chain_id1]:
                    # Not the tail of the chain
                    chain_id1 = -1
            else:
                chain_id1 = -1
            wait_for = t.wait_for
            if wait_for is not None:
                chain_id2 = wait_for.data.chain_id
                chain_pos2 = wait_for.chain_pos[chain_id2]
                if chain_pos2 + 1 != chain_lengths[chain_id2]:
                    # Not the tail of the chain
                    chain_id2 = -1
            else:
                chain_id2 = -1
            if chain_id1 == -1:
                if chain_id2 == -1:
                    # New chain
                    chain_id = chain_lengths.size()
                    chain_pos = 0
                    chain_lengths.push_back(0)
                else:
                    chain_id = chain_id2
                    chain_pos = chain_pos2 + 1
            elif chain_id2 == -1:
                chain_id = chain_id1
                chain_pos = chain_pos1 + 1
            elif chain_pos2 > chain_pos1:
                chain_id = chain_id2
                chain_pos = chain_pos2 + 1
            else:
                chain_id = chain_id1
                chain_pos = chain_pos1 + 1
            chain_lengths[chain_id] = chain_pos + 1
            t.data.chain_id = chain_id
            t.chain_pos.resize(chain_lengths.size(), -1)
            t.chain_pos[chain_id] = chain_pos

        cdef int nchains = chain_lengths.size()
        assume_not_none(event_times)
        for _t in event_times:
            t = <EventTime>_t
            t.chain_pos.resize(nchains, -1)
            prev = t.prev
            if prev is not None:
                update_chain_pos(t, prev, nchains)
            wait_for = t.wait_for
            if wait_for is not None:
                update_chain_pos(t, wait_for, nchains)

        return 0

    cdef long long compute_all_times(self, unsigned age) except -1:
        status = self.status.get()
        if not status.finalized:
            PyErr_Format(RuntimeError, "Event times not finalized")
        cdef long long max_time = 0
        event_times = self.event_times
        cdef int ntimes = PyList_GET_SIZE(event_times)
        self.time_values.resize(ntimes)
        cppfill(self.time_values.begin(), self.time_values.end(), -1)
        assume_not_none(event_times)
        for t in event_times:
            tv = get_time_value(<EventTime>t, -1, age, self.time_values)
            if tv > max_time:
                max_time = tv
        return max_time

cdef void update_chain_pos(EventTime self, EventTime prev, int nchains) noexcept:
    cdef int cid = self.data.chain_id
    for i in range(nchains):
        if i == cid:
            continue
        prev_pos = prev.chain_pos[i]
        if self.chain_pos[i] < prev_pos:
            self.chain_pos[i] = prev_pos

# If the base time has a static value, the returned time values will be the actual
# time point of the EventTime. If the base time does not have a static value
# the time offset relative to the base time is returned.
# This is so that if the base time is not statically known,
# we can compute the diff without computing the base time,
# while if the base time is known, we can use the static values in the computation
cdef long long get_time_value(EventTime self, int base_id, unsigned age,
                              vector[long long] &cache) except -1:
    cdef int tid = self.data.id
    if tid == base_id:
        if self.data.has_static:
            return self.data._get_static()
        return 0
    assert tid > base_id
    cdef long long value = cache[tid]
    if value >= 0:
        return value
    # The time manager should've been finalized and
    # no time should be floating anymore
    assert not self.data.floating

    if self.data.has_static:
        # If we have a static value it means that the base time has a static value
        # In this case, we are returning the full time and there's no need to
        # compute the offset from the base time.
        static_value = self.data._get_static()
        cache[tid] = static_value
        return static_value

    cdef EventTime prev = self.prev
    cdef long long prev_val = 0
    if prev is not None:
        prev_val = get_time_value(prev, base_id, age, cache)

    cdef bint cond = bool(get_value(self.cond, age))
    cdef long long offset = 0
    if cond:
        if self.rt_offset is not None:
            offset = <long long?>rt_eval(self.rt_offset, age)
            if offset < 0:
                bb_err_format(ValueError, event_time_key(<void*>self),
                              "Time delay cannot be negative")
        else:
            offset = <long long>self.data.c_offset

    cdef EventTime wait_for = self.wait_for
    if wait_for is None:
        # When wait_for is None, the offset is added to the previous time
        value = prev_val + offset
        cache[tid] = value
        return value

    # Otherwise, the wait_for is added to the wait_for time.
    value = prev_val
    cdef long long wait_for_val
    if cond:
        # Do not try to evaluate wait_for unless the condition is true
        # When a base_id is supplied, the wait_for event time may not share
        # this base if the condition isn't true.
        wait_for_val = get_time_value(wait_for, base_id, age, cache) + offset
        value = max(value, wait_for_val)
    cache[tid] = value
    return value

@cython.final
cdef class EventTime:
    def __init__(self):
        PyErr_Format(TypeError, "EventTime cannot be created directly")

    def __sub__(self, EventTime other):
        if self.manager_status.get() != other.manager_status.get():
            PyErr_Format(ValueError,
                         "Cannot take the difference between unrelated times")
        diff = <EventTimeDiff>EventTimeDiff.__new__(EventTimeDiff)
        diff.t1 = self
        diff.t2 = other
        diff.in_eval = False
        return new_extern_age(diff)

    def __str__(self):
        if self.data.floating:
            return '<floating>'
        if self.data.has_static:
            return _str_time(self.data._get_static())
        prev = self.prev
        if self.rt_offset is None:
            offset = _str_time(<long long>self.data.c_offset)
        else:
            offset = str(self.rt_offset)
        cond = self.cond
        if prev is None:
            assert cond is True
            return offset
        wait_for = self.wait_for
        if wait_for is None:
            if cond is True:
                return f'T[{prev.data.id}] + {offset}'
            return f'T[{prev.data.id}] + ({offset}; if {cond})'
        if cond is True:
            return f'T[{prev.data.id}]; wait_for(T[{wait_for.data.id}] + {offset})'
        return f'T[{prev.data.id}]; wait_for(T[{wait_for.data.id}] + {offset}; if {cond})'

    def __repr__(self):
        return str(self)


cdef TimeOrder is_ordered(EventTime t1, EventTime t2) except OrderError:
    assert t1.manager_status.get() == t2.manager_status.get()
    manager_status = t1.manager_status.get()
    if not manager_status.finalized:
        PyErr_Format(RuntimeError, "Event times not finalized")
    if t1 is t2:
        return OrderEqual
    chain1 = t1.data.chain_id
    chain2 = t2.data.chain_id
    idx1 = t1.chain_pos[chain1]
    idx2 = t2.chain_pos[chain1]
    if idx2 >= idx1:
        # Assume t1 and t2 are on different chain if idx1 == idx2
        # Since otherwise they should've been the same time
        return OrderBefore
    if chain1 == chain2:
        return OrderAfter
    idx1 = t1.chain_pos[chain2]
    idx2 = t2.chain_pos[chain2]
    if idx1 >= idx2:
        return OrderAfter
    return NoOrder

cdef EventTime find_common_root(EventTimeDiff self, unsigned age):
    cdef cppmap[int,void*] frontier
    if not self.t1.manager_status.get().finalized:
        PyErr_Format(RuntimeError, "Event times not finalized")
    frontier[self.t1.data.id] = <void*>self.t1
    frontier[self.t2.data.id] = <void*>self.t2
    while frontier.size() > 1:
        it = predec(frontier.end())
        t = <EventTime>deref(it).second
        frontier.erase(it)
        prev = t.prev
        if prev is None:
            return # Found the start of the experiment
        frontier[prev.data.id] = <void*>prev
        wait_for = t.wait_for
        if wait_for is not None:
            if not get_value(t.cond, age):
                continue
            frontier[wait_for.data.id] = <void*>wait_for
    return <EventTime>deref(frontier.begin()).second

cdef double timediff_eval(EventTimeDiff self, unsigned age) except? -100.0:
    cdef EventTime common_root = find_common_root(self, age)
    cdef int base_id = -1
    if common_root is not None:
        base_id = common_root.data.id
    t1 = self.t1
    t2 = self.t2
    cdef vector[long long] cache
    cache.resize(t1.manager_status.get().ntimes, -1)
    cdef double diff = get_time_value(t1, base_id, age, cache) - get_time_value(t2, base_id, age, cache)
    return diff / c_time_scale

@cython.final
cdef class EventTimeDiff:
    cdef EventTime t1
    cdef EventTime t2
    cdef bint in_eval

    def __init__(self):
        PyErr_Format(TypeError, "EventTimeDiff cannot be created directly")

    def __call__(self, unsigned age):
        if self.in_eval:
            PyErr_Format(ValueError, "Recursive value dependency detected.")
        self.in_eval = True
        try:
            return timediff_eval(self, age)
        finally:
            self.in_eval = False

    def __str__(self):
        return f'T[{self.t1.data.id}] - T[{self.t2.data.id}]'