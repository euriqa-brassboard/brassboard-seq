# cython: language_level=3

from cpython cimport PyDict_Size

cdef extern from *:
    """
struct ActionData {
    bool is_pulse;
    bool exact_time;
    bool cond_val;
};
    """
    cppclass ActionData:
        bint is_pulse;
        bint exact_time;
        bint cond_val;

cdef class Action:
    cdef object value
    cdef object cond
    cdef dict kws
    cdef ActionData data
    cdef int aid

    # Used for processing after the sequence has been programmed only.
    cdef int tid
    cdef int end_tid
    cdef object length
    cdef object prev_val
    cdef object end_val

cdef inline Action new_action(value, cond, bint is_pulse, bint exact_time, dict kws, int aid):
    self = <Action>Action.__new__(Action)
    self.value = value
    self.cond = cond
    self.data.is_pulse = is_pulse
    self.data.exact_time = exact_time
    self.kws = kws if PyDict_Size(kws) else None
    self.aid = aid
    return self
