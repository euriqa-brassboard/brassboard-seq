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

cdef class RampFunction:
    cdef dict params
    cdef object _eval

cdef ramp_eval(RampFunction self, t, length, oldval)
cdef int ramp_set_compile_params(RampFunction self) except -1
cdef int ramp_set_runtime_params(RampFunction self, unsigned age) except -1

cdef class RampBuffer:
    cdef object input_buff
    cdef object output_buff

cdef RampBuffer new_ramp_buffer()
cdef double *rampbuffer_alloc_input(self, int size) except NULL
cdef double *rampbuffer_eval(self, func, length, oldval) except NULL
