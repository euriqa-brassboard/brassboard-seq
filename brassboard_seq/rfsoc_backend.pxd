# cython: language_level=3

from libc.stdint cimport *

cdef extern from "src/rfsoc_backend.h" namespace "rfsoc_backend":
    struct cubic_spline_t:
        double order0
        double order1
        double order2
        double order3

    struct output_flags_t:
        bint wait_trigger
        bint sync
        bint feedback_enable

cdef class RFSOCOutputGenerator:
    cdef int start(self) except -1
    cdef int add_tone_data(self, int channel, int tone, int64_t duration_cycles,
                           cubic_spline_t frequency_hz, cubic_spline_t amplitude,
                           cubic_spline_t phase_rad, output_flags_t flags) except -1
    cdef int finish(self) except -1
