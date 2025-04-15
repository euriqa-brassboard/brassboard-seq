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
from brassboard_seq.event_time cimport new_time_manager, new_time_int, new_round_time
from brassboard_seq.rtval cimport is_rtval, RuntimeValue
from brassboard_seq.scan cimport new_empty_param_pack
from brassboard_seq.utils cimport assert_key, event_time_key, py_stringio, \
  PyErr_Format, PyExc_TypeError

cimport cython
from cpython cimport PyObject

cdef extern from "src/seq.cpp" namespace "brassboard_seq::seq":
    PyObject *timestep_type
    PyObject *subseq_type
    PyObject *condwrapper_type
    void update_timestep(TimeStep) except +
    void update_subseq(SubSeq, ConditionalWrapper, TimeSeq, TimeStep) except +
    void update_conditional(ConditionalWrapper, TimeSeq, TimeStep) except +
    int get_channel_id(SeqInfo self, str name) except +
    object combine_cond(object cond1, object new_cond) except +
    void timeseq_set_time(TimeSeq self, EventTime time, object offset) except +
    SubSeq add_custom_step(SubSeq, object cond, EventTime, object) except +


timestep_type = <PyObject*>TimeStep
subseq_type = <PyObject*>SubSeq
condwrapper_type = <PyObject*>ConditionalWrapper

update_timestep(None)
update_subseq(None, None, None, None)
update_conditional(None, None, None)

@cython.auto_pickle(False)
cdef class TimeSeq:
    def __init__(self):
        PyErr_Format(PyExc_TypeError, "TimeSeq cannot be created directly")

    def get_channel_id(self, str name, /):
        return get_channel_id(self.seqinfo, name)

    def set_time(self, EventTime time, /, offset=0): # offset in seconds
        timeseq_set_time(self, time, offset)

    def rt_assert(self, c, /, str msg="Assertion failed"):
        if is_rtval(c):
            self.seqinfo.bt_tracker.record(assert_key(len(self.seqinfo.assertions)))
            self.seqinfo.assertions.append((c, msg))
            return
        if not c:
            raise AssertionError(msg)

    @property
    def C(self):
        return self.seqinfo.C

@cython.auto_pickle(False)
@cython.no_gc
@cython.final
cdef class TimeStep(TimeSeq):
    def __init__(self):
        PyErr_Format(PyExc_TypeError, "TimeStep cannot be created directly")

    # Methods defined in c++
    # def set(self, chn, value, /, *, cond=True, bint exact_time=False, **kws)
    # def pulse(self, chn, value, /, *, cond=True, bint exact_time=False, **kws)

    def __str__(self):
        cdef py_stringio io
        timestep_show(self, io, 0)
        return io.getvalue()

    def __repr__(self):
        return str(self)

cdef int timestep_show(TimeStep self, py_stringio &io, int indent) except -1:
    io.write_rep_ascii(indent, b' ')
    io.write(f'TimeStep({self.length})@T[{self.start_time.data.id}]')
    cond = self.cond
    if cond is not True:
        io.write(f' if {cond}\n')
    else:
        io.write_ascii(b'\n')
    nactions = self.actions.size()
    for chn_idx in range(nactions):
        action = self.actions[chn_idx]
        if action == NULL:
            continue
        chn = '/'.join(self.seqinfo.channel_paths[chn_idx])
        io.write_rep_ascii(indent + 2, b' ')
        io.write(f'{chn}: {action.py_str()}\n')
    return 0

@cython.auto_pickle(False)
@cython.final
cdef class ConditionalWrapper:
    def __init__(self):
        PyErr_Format(PyExc_TypeError, "ConditionalWrapper cannot be created directly")

    def wait(self, length, /, *, cond=True):
        wait_cond(self.seq, length, combine_cond(self.cond, cond))

    def wait_for(self, tp, /, offset=0):
        wait_for_cond(self.seq, tp, offset, self.cond)

    # Methods defined in c++
    # def conditional(self, cond, /)
    # def add_step(self, first_arg, /, *args, **kwargs)
    # def add_background(self, first_arg, /, *args, **kwargs)
    # def add_floating(self, first_arg, /, *args, **kwargs)
    # def add_at(self, EventTime tp, first_arg, /, *args, **kwargs)
    # def set(self, chn, value, /, *, cond=True, bint exact_time=False, **kws)

    # Shorthand for add_step of custom step. Meant to be used as decorator
    def __call__(self, cb, /):
        step = add_custom_step(self.seq, self.cond, self.seq.end_time, cb)
        self.seq.end_time = step.end_time
        return step

    def __str__(self):
        cdef py_stringio io
        conditionalwrapper_show(self, io, 0)
        return io.getvalue()

    def __repr__(self):
        return str(self)

    @property
    def C(self):
        return self.seq.seqinfo.C

cdef int conditionalwrapper_show(ConditionalWrapper self, py_stringio &io, int indent) except -1:
    io.write_rep_ascii(indent, b' ')
    io.write(f'ConditionalWrapper({self.cond}) for\n')
    if type(self.seq) == Seq:
        return seq_show(self.seq, io, indent + 2)
    return subseq_show(self.seq, io, indent + 2)

@cython.auto_pickle(False)
cdef class SubSeq(TimeSeq):
    def __init__(self):
        PyErr_Format(PyExc_TypeError, "SubSeq cannot be created directly")

    @property
    def current_time(self):
        return self.end_time

    def wait(self, length, /, *, cond=True):
        wait_cond(self, length, combine_cond(self.cond, cond))

    def wait_for(self, tp, /, offset=0):
        wait_for_cond(self, tp, offset, self.cond)

    # Methods defined in c++
    # def conditional(self, cond, /)
    # def add_step(self, first_arg, /, *args, **kwargs)
    # def add_background(self, first_arg, /, *args, **kwargs)
    # def add_floating(self, first_arg, /, *args, **kwargs)
    # def add_at(self, EventTime tp, first_arg, /, *args, **kwargs)
    # def set(self, chn, value, /, *, cond=True, bint exact_time=False, **kws)

    def __str__(self):
        cdef py_stringio io
        subseq_show(self, io, 0)
        return io.getvalue()

    def __repr__(self):
        return str(self)

cdef int wait_cond(SubSeq self, length, cond) except -1:
    self.end_time = new_round_time(self.seqinfo.time_mgr, self.end_time, length,
                                   cond, None)
    self.seqinfo.bt_tracker.record(event_time_key(<void*>self.end_time))
    return 0

cdef int wait_for_cond(SubSeq self, _tp0, offset, cond) except -1:
    cdef EventTime tp0
    if type(_tp0) is EventTime:
        tp0 = <EventTime>_tp0
    else:
        tp0 = (<TimeSeq?>_tp0).end_time
    self.end_time = new_round_time(self.seqinfo.time_mgr, self.end_time, offset,
                                   cond, tp0)
    self.seqinfo.bt_tracker.record(event_time_key(<void*>self.end_time))
    return 0

cdef int subseq_show_subseqs(SubSeq self, py_stringio &io, int indent) except -1:
    for _subseq in self.sub_seqs:
        subseq = <TimeSeq>_subseq
        if type(subseq) is TimeStep:
            timestep_show(<TimeStep>subseq, io, indent)
        else:
            subseq_show(<SubSeq>subseq, io, indent)
    return 0

cdef int subseq_show(SubSeq self, py_stringio &io, int indent) except -1:
    io.write_rep_ascii(indent, b' ')
    io.write(f'SubSeq@T[{self.start_time.data.id}] - T[{self.end_time.data.id}]')
    cond = self.cond
    if cond is not True:
        io.write(f' if {cond}\n')
    else:
        io.write_ascii(b'\n')
    subseq_show_subseqs(self, io, indent + 2)
    return 0

@cython.auto_pickle(False)
@cython.no_gc
@cython.final
cdef class SeqInfo:
    def __init__(self):
        PyErr_Format(PyExc_TypeError, "SeqInfo cannot be created directly")

@cython.auto_pickle(False)
@cython.final
cdef class Seq(SubSeq):
    def __init__(self, Config config, /, int max_frame=0):
        self.cond = True
        self.sub_seqs = []
        seqinfo = <SeqInfo>SeqInfo.__new__(SeqInfo)
        seqinfo.config = config
        seqinfo.time_mgr = new_time_manager()
        seqinfo.assertions = []
        seqinfo.bt_tracker.max_frame = max_frame
        seqinfo.channel_name_map = {}
        seqinfo.channel_path_map = {}
        seqinfo.channel_paths = []
        seqinfo.C = new_empty_param_pack()
        seqinfo.action_counter = 0
        self.seqinfo = seqinfo
        self.end_time = new_time_int(seqinfo.time_mgr, None, 0, False, True, None)
        self.seqinfo.bt_tracker.record(event_time_key(<void*>self.end_time))

    def __str__(self):
        cdef py_stringio io
        seq_show(self, io, 0)
        return io.getvalue()

    def __repr__(self):
        return str(self)

cdef int seq_show(Seq self, py_stringio &io, int indent) except -1:
    io.write_rep_ascii(indent, b' ')
    io.write(f'Seq - T[{self.end_time.data.id}]\n')
    cdef int i = 0
    for t in self.seqinfo.time_mgr.event_times:
        io.write_rep_ascii(indent + 1, b' ')
        io.write(f'T[{i}]: ')
        io.write(str(t))
        io.write_ascii('\n')
        i += 1
    return subseq_show_subseqs(self, io, indent + 2)
