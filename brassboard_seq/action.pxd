# cython: language_level=3

from cpython cimport PyObject

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

cdef class RampFunction:
    cdef dict params

    cdef int set_compile_params(self) except -1
    cdef int set_runtime_params(self, long long age) except -1

    cpdef eval(self, t, length, oldval)

cdef class RampBuffer:
    cdef object input_buff
    cdef object output_buff
    cdef object exception

    cdef double *alloc_input(self, int size) except NULL
    cdef double *eval(self, RampFunction func, double length, double oldval) except NULL

cdef RampBuffer new_ramp_buffer()
cdef double *rampbuffer_alloc_input(PyObject *self, int size) noexcept
cdef double *rampbuffer_eval(PyObject *self, PyObject *func,
                             double length, double oldval) noexcept
