# cython: language_level=3

# Do not use relative import since it messes up cython file name tracking
from brassboard_seq.config cimport Config
from brassboard_seq.event_time cimport TimeManager, EventTime, new_time_manager

cdef class TimeSeq:
    # Toplevel parent sequence
    cdef SeqInfo seqinfo
    # The starting time of this sequence/step
    # This time point may or may not be related to the start time
    # of the parent sequence.
    cdef readonly EventTime start_time
    # Ending time, for SubSeq this is also the time the next step is added by default
    cdef readonly EventTime end_time
    # Condition for this sequence/step to be enabled.
    # This can be either a runtime value or `True` or `False`.
    # This is also always guaranteed to be true only if the parent's condition is true.
    cdef object cond
    # A LLVM style flag for type checking.
    cdef bint is_step


cdef class TimeStep(TimeSeq):
    # This is the length that was passed in by the user (in unit of second)
    # to create the step without considering the condition if this step is enabled.
    # This is also the length parameter that'll be passed to the user function
    # if the action added to this step contains ramps.
    cdef object length
    # The dict of channel -> actions
    cdef dict actions

    cdef int _set(self, chn, value, cond, bint is_pulse, bint exact_time, dict kws) except -1


cdef class ConditionalWrapper:
    cdef SubSeq seq
    cdef object cond

    cpdef ConditionalWrapper conditional(self, cond)


cdef class SubSeq(TimeSeq):
    # The list of subsequences and steps in this subsequcne
    cdef list sub_seqs
    cdef TimeStep dummy_step

    cpdef ConditionalWrapper conditional(self, cond)

    cdef int wait_cond(self, length, cond) except -1 # length in seconds
    cdef SubSeq add_custom_step(self, cond, EventTime start_time, cb,
                                tuple args, dict kwargs)
    # length in seconds
    cdef TimeStep add_time_step(self, cond, EventTime start_time, length)
    cdef TimeSeq add_step_real(self, cond, EventTime start_time,
                               first_arg, tuple args, dict kwargs)

    cdef int wait_for_cond(self, tp, offset, cond) except -1 # offset in seconds

    cdef int _set(self, chn, value, cond, bint exact_time, dict kws) except -1

    cdef int collect_actions(self, list actions) except -1


cdef class SeqInfo:
    # EventTime manager
    cdef TimeManager time_mgr
    # Global config object
    cdef Config config
    # Name<->channel ID mapping
    cdef dict channel_name_map
    cdef dict channel_path_map
    cdef list channel_paths

    cdef int _get_channel_id(self, str name) except -1


cdef class Seq(SubSeq):
    cdef list all_actions

    cdef int finalize(self) except -1

cpdef Seq new_seq(Config config)
