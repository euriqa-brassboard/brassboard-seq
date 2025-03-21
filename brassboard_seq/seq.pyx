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
from brassboard_seq.action cimport action_str
from brassboard_seq.event_time cimport round_time_int, round_time_rt, \
  set_base_int, set_base_rt, rt_time_scale, new_time_manager
from brassboard_seq.rtval cimport is_rtval, RuntimeValue
from brassboard_seq.scan cimport new_param_pack
from brassboard_seq.utils cimport assert_key, event_time_key, py_stringio, \
  PyErr_Format, PyExc_TypeError

cimport cython
from cpython cimport PyObject, PyList_GET_SIZE

cdef extern from "src/seq.cpp" namespace "brassboard_seq::seq":
    PyObject *event_time_type
    PyObject *timestep_type
    PyObject *subseq_type
    PyObject *condwrapper_type
    PyObject *_rt_time_scale "brassboard_seq::seq::rt_time_scale"
    void update_timestep(TimeStep) except +
    void update_subseq(SubSeq, ConditionalWrapper, TimeSeq, TimeStep) except +
    void update_conditional(ConditionalWrapper, TimeSeq, TimeStep) except +
    int get_channel_id(SeqInfo self, str name) except +
    object combine_cond(object cond1, object new_cond) except +
    SubSeq add_custom_step(SubSeq, object cond, EventTime, object) except +
    void basicseq_add_branch(BasicSeq self, BasicSeq bseq) except +
    bint basicseq_may_terminate(BasicSeq self)


event_time_type = <PyObject*>EventTime
timestep_type = <PyObject*>TimeStep
subseq_type = <PyObject*>SubSeq
condwrapper_type = <PyObject*>ConditionalWrapper
_rt_time_scale = <PyObject*>rt_time_scale

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
        if is_rtval(offset):
            set_base_rt(self.start_time, time,
                        round_time_rt(<RuntimeValue>offset, rt_time_scale))
        else:
            set_base_int(self.start_time, time, round_time_int(offset))
        self.seqinfo.cinfo.bt_tracker.record(event_time_key(<void*>self.start_time))

    def rt_assert(self, c, /, str msg="Assertion failed"):
        if is_rtval(c):
            self.seqinfo.cinfo.bt_tracker.record(assert_key(len(self.seqinfo.assertions)))
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
        io.write(f'{chn}: {action_str(action)}\n')
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
    self.end_time = self.seqinfo.time_mgr.new_round_time(self.end_time, length,
                                                         cond, None)
    self.seqinfo.cinfo.bt_tracker.record(event_time_key(<void*>self.end_time))
    return 0

cdef int wait_for_cond(SubSeq self, _tp0, offset, cond) except -1:
    cdef EventTime tp0
    if type(_tp0) is EventTime:
        tp0 = <EventTime>_tp0
    else:
        tp0 = (<TimeSeq?>_tp0).end_time
    self.end_time = self.seqinfo.time_mgr.new_round_time(self.end_time, offset,
                                                         cond, tp0)
    self.seqinfo.cinfo.bt_tracker.record(event_time_key(<void*>self.end_time))
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

cdef int basicseq_show_next(BasicSeq self, py_stringio &io, int indent) except -1:
    if self.next_bseq.empty():
        return 0
    io.write_rep_ascii(indent, b' ')
    io.write_ascii(b'branches: [')
    cdef size_t i
    for i in range(self.next_bseq.size()):
        bseq_id = self.next_bseq[i]
        if i != 0:
            io.write_ascii(b', ')
        io.write(f'{bseq_id}')
    io.write_ascii(b']')
    if basicseq_may_terminate(self):
        io.write_ascii(b' may terminate')
    io.write_ascii(b'\n')

cdef int basicseq_show(BasicSeq self, py_stringio &io, int indent) except -1:
    io.write_rep_ascii(indent, b' ')
    io.write(f'BasicSeq[{(<BasicSeq>self).bseq_id}]')
    io.write(f' - T[{self.end_time.data.id}]\n')
    basicseq_show_next(self, io, indent + 1)
    cdef int i = 0
    for t in self.seqinfo.time_mgr.event_times:
        io.write_rep_ascii(indent + 1, b' ')
        io.write(f'T[{i}]: ')
        io.write(str(t))
        io.write_ascii(b'\n')
        i += 1
    subseq_show_subseqs(self, io, indent + 2)
    return 0

@cython.auto_pickle(False)
@cython.no_gc
@cython.final
cdef class SeqInfo:
    def __init__(self):
        PyErr_Format(PyExc_TypeError, "SeqInfo cannot be created directly")

@cython.auto_pickle(False)
cdef class BasicSeq(SubSeq):
    def __init__(self):
        PyErr_Format(PyExc_TypeError, "BasicSeq cannot be created directly")

    def new_basic_seq(self):
        new_bseq = <BasicSeq>BasicSeq.__new__(BasicSeq)
        new_bseq.cond = True
        new_bseq.term_status = TerminateStatus.Default
        new_bseq.sub_seqs = []
        new_bseq.basic_seqs = self.basic_seqs
        new_bseq.bseq_id = len(self.basic_seqs)
        self.basic_seqs.append(new_bseq)

        new_seqinfo = <SeqInfo>SeqInfo.__new__(SeqInfo)
        new_seqinfo.time_mgr = new_time_manager()
        new_bseq.seqinfo = new_seqinfo

        seqinfo = self.seqinfo
        new_seqinfo.config = seqinfo.config
        new_seqinfo.assertions = seqinfo.assertions
        new_seqinfo.channel_name_map = seqinfo.channel_name_map
        new_seqinfo.channel_path_map = seqinfo.channel_path_map
        new_seqinfo.channel_paths = seqinfo.channel_paths
        new_seqinfo.C = seqinfo.C
        new_seqinfo.cinfo = seqinfo.cinfo
        new_bseq.end_time = new_seqinfo.time_mgr.new_time_int(None, 0, False, True, None)
        seqinfo.cinfo.bt_tracker.record(event_time_key(<void*>new_bseq.end_time))

        return new_bseq

    @property
    def may_terminate(self):
        return basicseq_may_terminate(self)

    @may_terminate.setter
    def may_terminate(self, bint may_term):
        self.term_status = (TerminateStatus.MayTerm if may_term
                            else TerminateStatus.MayNotTerm)

    def add_branch(self, BasicSeq bseq):
        basicseq_add_branch(self, bseq)

    def __str__(self):
        cdef py_stringio io
        basicseq_show(self, io, 0)
        return io.getvalue()

    def __repr__(self):
        return str(self)

@cython.auto_pickle(False)
@cython.final
cdef class Seq(BasicSeq):
    def __init__(self, Config config, /, int max_frame=0):
        self.bseq_id = 0
        self.term_status = TerminateStatus.Default
        self.cond = True
        self.sub_seqs = []
        self.basic_seqs = [self]
        cinfo = <CInfo>CInfo.__new__(CInfo)
        cinfo.bt_tracker.max_frame = max_frame
        cinfo.action_counter = 0
        seqinfo = <SeqInfo>SeqInfo.__new__(SeqInfo)
        seqinfo.config = config
        seqinfo.time_mgr = new_time_manager()
        seqinfo.assertions = []
        seqinfo.channel_name_map = {}
        seqinfo.channel_path_map = {}
        seqinfo.channel_paths = []
        seqinfo.C = new_param_pack(ParamPack, {}, {}, 'root', None)
        seqinfo.cinfo = cinfo
        self.seqinfo = seqinfo
        self.end_time = seqinfo.time_mgr.new_time_int(None, 0, False, True, None)
        cinfo.bt_tracker.record(event_time_key(<void*>self.end_time))

    def __str__(self):
        cdef py_stringio io
        seq_show(self, io, 0)
        return io.getvalue()

    def __repr__(self):
        return str(self)

cdef int seq_show(Seq self, py_stringio &io, int indent) except -1:
    io.write_rep_ascii(indent, b' ')
    io.write(f'Seq - T[{self.end_time.data.id}]\n')
    if PyList_GET_SIZE(self.basic_seqs) > 1:
        basicseq_show_next(self, io, indent + 1)
    cdef int i = 0
    for t in self.seqinfo.time_mgr.event_times:
        io.write_rep_ascii(indent + 1, b' ')
        io.write(f'T[{i}]: ')
        io.write(str(t))
        io.write_ascii('\n')
        i += 1
    subseq_show_subseqs(self, io, indent + 2)
    for i in range(1, PyList_GET_SIZE(self.basic_seqs)):
        io.write_ascii(b'\n')
        basicseq_show(<BasicSeq>self.basic_seqs[i], io, indent + 1)
