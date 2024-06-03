# cython: language_level=3

# Do not use relative import since it messes up cython file name tracking
from brassboard_seq.rtval cimport ifelse, get_value, max_val, \
  new_const, new_extern_age

from libcpp.map cimport map as cppmap
from libcpp.algorithm cimport fill as cppfill

cimport cython
from cython.operator cimport dereference as deref, predecrement as predec

cdef object py_time_scale = c_time_scale
cdef RuntimeValue rt_time_scale = new_const(py_time_scale)

def time_scale():
    return py_time_scale

cdef str _str_time(t):
    if is_rtval(t):
        return str(<RuntimeValue>t)
    s = <double>c_time_scale
    ms = c_time_scale / 1e3
    us = c_time_scale / 1e6
    ns = c_time_scale / 1e9
    ps = c_time_scale / 1e12

    if t >= 0.3 * s:
        return f'{t / s:.12} s'
    if t >= 0.3 * ms:
        return f'{t / ms:.9} ms'
    if t >= 0.3 * us:
        return f'{t / us:.6} us'
    if t >= 0.3 * ns:
        return f'{t / ns:.3} ns'
    return f'{t / ps} ps'

@cython.final
cdef class TimeManager:
    def __init__(self):
        PyErr_Format(TypeError, "TimeManager cannot be created directly")

    cdef int visit_time(self, EventTime t, unordered_set[int] &visited) except -1:
        cdef int id = t.id
        if id < 0:
            return 0
        if t.floating:
            PyErr_Format(RuntimeError, "Event time still floating")
        if visited.count(id):
            PyErr_Format(ValueError, "Time loop detected")
        visited.insert(id)
        prev = t.prev
        if prev is not None:
            self.visit_time(prev, visited)
        wait_for = t.wait_for
        if wait_for is not None:
            self.visit_time(wait_for, visited)
        t.id = -1
        self.event_times.append(t)
        return 0

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
        for _t in old_event_times:
            self.visit_time(<EventTime>_t, visited)

        cdef vector[int] chain_lengths
        cdef int i = 0
        cdef int chain_id1
        cdef int chain_pos1
        cdef int chain_id2
        cdef int chain_pos2
        cdef int chain_id
        cdef int chain_pos
        cdef EventTime t
        for _t in event_times:
            t = <EventTime>_t
            tid = i
            i += 1
            t.id = tid
            prev = t.prev
            if prev is not None:
                chain_id1 = prev.chain_id
                chain_pos1 = prev.chain_pos[chain_id1]
                if chain_pos1 + 1 != chain_lengths[chain_id1]:
                    # Not the tail of the chain
                    chain_id1 = -1
            else:
                chain_id1 = -1
            wait_for = t.wait_for
            if wait_for is not None:
                chain_id2 = wait_for.chain_id
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
            t.chain_id = chain_id
            t.chain_pos.resize(chain_lengths.size(), -1)
            t.chain_pos[chain_id] = chain_pos

        cdef int nchains = chain_lengths.size()
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

    cdef long long compute_all_times(self, long long age) except -1:
        status = self.status.get()
        if not status.finalized:
            PyErr_Format(RuntimeError, "Event times not finalized")
        cdef long long max_time = 0
        event_times = self.event_times
        cdef int ntimes = PyList_GET_SIZE(event_times)
        self.time_values.resize(ntimes)
        cppfill(self.time_values.begin(), self.time_values.end(), -1)
        for t in event_times:
            tv = (<EventTime>t).get_value(-1, age, self.time_values)
            if tv > max_time:
                max_time = tv
        return max_time

cdef void update_chain_pos(EventTime self, EventTime prev, int nchains) noexcept:
    cdef int cid = self.chain_id
    for i in range(nchains):
        if i == cid:
            continue
        prev_pos = prev.chain_pos[i]
        if self.chain_pos[i] < prev_pos:
            self.chain_pos[i] = prev_pos

@cython.final
cdef class EventTime:
    def __init__(self):
        PyErr_Format(TypeError, "EventTime cannot be created directly")

    cdef int set_base_int(self, EventTime base, long long offset) except -1:
        if not self.floating:
            PyErr_Format(ValueError, "Cannot modify non-floating time")
        self.prev = base
        self.c_offset = <long long>offset
        self.floating = False
        return 0

    cdef int set_base_rt(self, EventTime base, RuntimeValue offset) except -1:
        if not self.floating:
            PyErr_Format(ValueError, "Cannot modify non-floating time")
        self.prev = base
        self.rt_offset = offset
        self.floating = False
        return 0

    cdef long long get_value(self, int base_id, long long age,
                             vector[long long] &cache) except -1:
        cdef int tid = self.id
        if tid == base_id:
            return 0
        assert tid > base_id
        cdef long long value = cache[tid]
        if value >= 0:
            return value
        # The time manager should've been finalized and
        # no time should be floating anymore
        assert not self.floating

        cdef EventTime prev = self.prev
        cdef long long prev_val = 0
        if prev is not None:
            prev_val = prev.get_value(base_id, age, cache)

        cdef bint cond = bool(get_value(self.cond, age))
        cdef long long offset = 0
        if cond:
            if self.rt_offset is not None:
                offset = <long long?>get_value(self.rt_offset, age)
            else:
                offset = self.c_offset
            if offset < 0:
                # TODO: sequence backtrace
                PyErr_Format(ValueError, "Time delay cannot be negative")


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
            wait_for_val = wait_for.get_value(base_id, age, cache) + offset
            value = max(value, wait_for_val)
        cache[tid] = value
        return value

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
        if self.floating:
            return '<floating>'
        prev = self.prev
        offset = _str_time(self.c_offset if self.rt_offset is None else self.rt_offset)
        cond = self.cond
        if prev is None:
            assert cond is True
            return offset
        wait_for = self.wait_for
        if wait_for is None:
            if cond is True:
                return f'T[{prev.id}] + {offset}'
            return f'T[{prev.id}] + ({offset}; if {cond})'
        if cond is True:
            return f'T[{prev.id}]; wait_for(T[{wait_for.id}] + {offset})'
        return f'T[{prev.id}]; wait_for(T[{wait_for.id}] + {offset}; if {cond})'

    def __repr__(self):
        return str(self)


cdef TimeOrder is_ordered(EventTime t1, EventTime t2) except OrderError:
    assert t1.manager_status.get() == t2.manager_status.get()
    manager_status = t1.manager_status.get()
    if not manager_status.finalized:
        PyErr_Format(RuntimeError, "Event times not finalized")
    if t1 is t2:
        return OrderEqual
    chain1 = t1.chain_id
    chain2 = t2.chain_id
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


@cython.final
cdef class EventTimeDiff:
    cdef EventTime t1
    cdef EventTime t2
    cdef bint in_eval

    def __init__(self):
        PyErr_Format(TypeError, "EventTimeDiff cannot be created directly")

    cdef EventTime find_common_root(self, long long age):
        cdef cppmap[int,void*] frontier
        if not self.t1.manager_status.get().finalized:
            PyErr_Format(RuntimeError, "Event times not finalized")
        frontier[self.t1.id] = <void*>self.t1
        frontier[self.t2.id] = <void*>self.t2
        while frontier.size() > 1:
            it = predec(frontier.end())
            t = <EventTime>deref(it).second
            frontier.erase(it)
            prev = t.prev
            if prev is None:
                return # Found the start of the experiment
            frontier[prev.id] = <void*>prev
            wait_for = t.wait_for
            if wait_for is not None:
                if not get_value(t.cond, age):
                    continue
                frontier[wait_for.id] = <void*>wait_for
        return <EventTime>deref(frontier.begin()).second

    cdef double _eval(self, long long age):
        cdef EventTime common_root = self.find_common_root(age)
        cdef int base_id = -1
        if common_root is not None:
            base_id = common_root.id
        t1 = self.t1
        t2 = self.t2
        cdef vector[long long] cache
        cache.resize(t1.manager_status.get().ntimes, -1)
        cdef double diff = t1.get_value(base_id, age, cache) - t2.get_value(base_id, age, cache)
        return diff / c_time_scale

    def __call__(self, long long age):
        if self.in_eval:
            PyErr_Format(ValueError, "Recursive value dependency detected.")
        self.in_eval = True
        try:
            return self._eval(age)
        finally:
            self.in_eval = False

    def __str__(self):
        return f'T[{self.t1.id}] - T[{self.t2.id}]'
