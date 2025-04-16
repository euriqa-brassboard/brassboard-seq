# cython: language_level=3

# Copyright (c) 2024 - 2025 Yichao Yu <yyc1992@gmail.com>

# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 3.0 of the License, or (at your option) any later version.

# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public
# License along with this library. If not,
# see <http://www.gnu.org/licenses/>.

# Do not use relative import since it messes up cython file name tracking
from brassboard_seq.config cimport Config
from brassboard_seq.event_time cimport TimeManager, EventTime
from brassboard_seq.utils cimport BacktraceTracker
from brassboard_seq.scan cimport ParamPack
from brassboard_seq.action cimport Action, ActionAllocator

from libcpp.vector cimport vector

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


cdef class TimeStep(TimeSeq):
    # This is the length that was passed in by the user (in unit of second)
    # to create the step without considering the condition if this step is enabled.
    # This is also the length parameter that'll be passed to the user function
    # if the action added to this step contains ramps.
    cdef object length
    # The array of channel -> actions
    cdef vector[Action*] actions


cdef class ConditionalWrapper:
    cdef SubSeq seq
    cdef object cond
    cdef void *fptr


cdef class SubSeq(TimeSeq):
    # The list of subsequences and steps in this subsequcne
    cdef list sub_seqs
    cdef TimeStep dummy_step


cdef class SeqInfo:
    # EventTime manager
    cdef TimeManager time_mgr
    # Global assumptions
    cdef list assertions
    # Global config object
    cdef Config config
    # Backtrace collection
    cdef BacktraceTracker bt_tracker
    # Name<->channel ID mapping
    cdef dict channel_name_map
    cdef dict channel_path_map
    cdef list channel_paths
    cdef ParamPack C
    cdef ActionAllocator action_alloc
    cdef int action_counter

cdef class Seq(SubSeq):
    pass
