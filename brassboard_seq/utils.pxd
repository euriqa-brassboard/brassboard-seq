# cython: language_level=3

from cpython cimport PyObject

from libc.stdint cimport *

cdef extern from "src/utils.h" namespace "brassboard_seq":
    T assume[T](T) noexcept nogil
    void assume_not_none(object) noexcept nogil
    void _assume_not_none "brassboard_seq::assume_not_none"(void*) noexcept nogil

    cppclass BacktraceTracker:
        int max_frame
        void record(uintptr_t key)
        object get_backtrace(uintptr_t key)

        cppclass GlobalRestorer:
            pass

    BacktraceTracker.GlobalRestorer set_global_tracker(BacktraceTracker*)

    uintptr_t event_time_key(void*)
    uintptr_t action_key(int)
    uintptr_t assert_key(int)

    int bb_raise(object exc, uintptr_t key) except 0
    int bb_err_format(object exc, uintptr_t key, const char *format, ...) except 0

    object pyfloat_from_double(double v)
    list new_list_of_list(int n)
    object pynum_add_or_sub(object a, object b, bint issub)
