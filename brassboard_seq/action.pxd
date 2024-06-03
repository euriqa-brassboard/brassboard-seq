# cython: language_level=3

cdef class Action:
    cdef object value
    cdef object cond
    cdef bint is_pulse
    cdef bint exact_time
    cdef dict kws

    # Used for processing after the sequence has been programmed only.
    cdef int tid
    cdef int end_tid
    cdef object length
    cdef object prev_val
    cdef object end_val

    cdef bint cond_val

cdef inline Action new_action(value, cond, bint is_pulse, bint exact_time, dict kws):
    self = <Action>Action.__new__(Action)
    self.value = value
    self.cond = cond
    self.is_pulse = is_pulse
    self.exact_time = exact_time
    self.kws = kws
    return self
