# cython: language_level=3

# Copyright (c) 2024 - 2024 Yichao Yu <yyc1992@gmail.com>

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
from brassboard_seq.action cimport RampFunction, action_str
from brassboard_seq.config cimport translate_channel
from brassboard_seq.event_time cimport round_time_int, round_time_rt, \
  set_base_int, set_base_rt, rt_time_scale, new_time_manager
from brassboard_seq.rtval cimport is_rtval, RuntimeValue
from brassboard_seq.scan cimport new_param_pack
from brassboard_seq.utils cimport assume_not_none, _assume_not_none, \
  action_key, assert_key, event_time_key, \
  PyErr_Format, PyExc_TypeError, PyExc_ValueError

cdef StringIO # hide import
from io import StringIO

cimport cython
from cpython cimport PyObject, PyDict_GetItemWithError, PyList_GET_SIZE, PyTypeObject

cdef extern from "src/seq.cpp" namespace "brassboard_seq::seq":
    PyTypeObject *event_time_type
    PyTypeObject *runtime_value_type
    PyTypeObject *timestep_type
    PyTypeObject *subseq_type
    PyTypeObject *condwrapper_type
    PyTypeObject *rampfunction_type
    PyObject *_rt_time_scale "brassboard_seq::seq::rt_time_scale"
    void update_timestep(TimeStep, RuntimeValue) except +
    void update_subseq(SubSeq, ConditionalWrapper, TimeSeq, TimeStep,
                       RuntimeValue) except +
    void update_conditional(ConditionalWrapper, TimeSeq, TimeStep,
                            RuntimeValue) except +
    object combine_cond(object cond1, object new_cond, RuntimeValue) except +
    SubSeq add_custom_step(SubSeq, object cond, EventTime, object,
                           RuntimeValue) except +
    void seq_finalize(Seq, TimeStep, RampFunction, RuntimeValue) except +
    void seq_runtime_finalize(Seq, unsigned age, py_object &pyage,
                              RampFunction, RuntimeValue) except +


event_time_type = <PyTypeObject*>EventTime
runtime_value_type = <PyTypeObject*>RuntimeValue
timestep_type = <PyTypeObject*>TimeStep
subseq_type = <PyTypeObject*>SubSeq
condwrapper_type = <PyTypeObject*>ConditionalWrapper
rampfunction_type = <PyTypeObject*>RampFunction
_rt_time_scale = <PyObject*>rt_time_scale

update_timestep(None, None)
update_subseq(None, None, None, None, None)
update_conditional(None, None, None, None)

@cython.auto_pickle(False)
cdef class TimeSeq:
    def __init__(self):
        PyErr_Format(PyExc_TypeError, "TimeSeq cannot be created directly")

    def get_channel_id(self, str name, /):
        return _get_channel_id(self.seqinfo, name)

    def set_time(self, EventTime time, /, offset=0): # offset in seconds
        if is_rtval(offset):
            set_base_rt(self.start_time, time,
                        round_time_rt(RuntimeValue, <RuntimeValue>offset,
                                      rt_time_scale))
        else:
            set_base_int(self.start_time, time, round_time_int(offset))
        self.seqinfo.bt_tracker.record(event_time_key(<void*>self.start_time))

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
        io = StringIO()
        timestep_show(self, io.write, 0)
        return io.getvalue()

    def __repr__(self):
        return str(self)

cdef int timestep_show(TimeStep self, write, int indent) except -1:
    write(' ' * indent)
    write(f'TimeStep({self.length})@T[{self.start_time.data.id}]')
    cond = self.cond
    if cond is not True:
        write(f' if {cond}\n')
    else:
        write('\n')
    nactions = self.actions.size()
    for chn_idx in range(nactions):
        action = self.actions[chn_idx]
        if action == NULL:
            continue
        chn = '/'.join(self.seqinfo.channel_paths[chn_idx])
        write(' ' * (indent + 2))
        write(f'{chn}: {action_str(action)}\n')
    return 0

@cython.auto_pickle(False)
@cython.final
cdef class ConditionalWrapper:
    def __init__(self):
        PyErr_Format(PyExc_TypeError, "ConditionalWrapper cannot be created directly")

    def wait(self, length, /, *, cond=True):
        wait_cond(self.seq, length, combine_cond(self.cond, cond, None))

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
        step = add_custom_step(self.seq, self.cond, self.seq.end_time, cb, None)
        self.seq.end_time = step.end_time
        return step

    def __str__(self):
        io = StringIO()
        conditionalwrapper_show(self, io.write, 0)
        return io.getvalue()

    def __repr__(self):
        return str(self)

    @property
    def C(self):
        return self.seq.seqinfo.C

cdef int conditionalwrapper_show(ConditionalWrapper self, write, int indent) except -1:
    write(' ' * indent)
    write(f'ConditionalWrapper({self.cond}) for\n')
    if type(self.seq) == Seq:
        return seq_show(self.seq, write, indent + 2)
    return subseq_show(self.seq, write, indent + 2)

@cython.auto_pickle(False)
cdef class SubSeq(TimeSeq):
    def __init__(self):
        PyErr_Format(PyExc_TypeError, "SubSeq cannot be created directly")

    @property
    def current_time(self):
        return self.end_time

    def wait(self, length, /, *, cond=True):
        wait_cond(self, length, combine_cond(self.cond, cond, None))

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
        io = StringIO()
        subseq_show(self, io.write, 0)
        return io.getvalue()

    def __repr__(self):
        return str(self)

cdef int wait_cond(SubSeq self, length, cond) except -1:
    self.end_time = self.seqinfo.time_mgr.new_round_time(self.end_time, length,
                                                         cond, None)
    self.seqinfo.bt_tracker.record(event_time_key(<void*>self.end_time))
    return 0

cdef int wait_for_cond(SubSeq self, _tp0, offset, cond) except -1:
    cdef EventTime tp0
    if type(_tp0) is EventTime:
        tp0 = <EventTime>_tp0
    else:
        tp0 = (<TimeSeq?>_tp0).end_time
    self.end_time = self.seqinfo.time_mgr.new_round_time(self.end_time, offset,
                                                         cond, tp0)
    self.seqinfo.bt_tracker.record(event_time_key(<void*>self.end_time))
    return 0

cdef int subseq_show_subseqs(SubSeq self, write, int indent) except -1:
    for _subseq in self.sub_seqs:
        subseq = <TimeSeq>_subseq
        if type(subseq) is TimeStep:
            timestep_show(<TimeStep>subseq, write, indent)
        else:
            subseq_show(<SubSeq>subseq, write, indent)
    return 0

cdef int subseq_show(SubSeq self, write, int indent) except -1:
    write(' ' * indent)
    write(f'SubSeq@T[{self.start_time.data.id}] - T[{self.end_time.data.id}]')
    cond = self.cond
    if cond is not True:
        write(f' if {cond}\n')
    else:
        write('\n')
    subseq_show_subseqs(self, write, indent + 2)
    return 0

@cython.auto_pickle(False)
@cython.no_gc
@cython.final
cdef class SeqInfo:
    def __init__(self):
        PyErr_Format(PyExc_TypeError, "SeqInfo cannot be created directly")

cdef int _get_channel_id(SeqInfo self, str name) except -1:
    channel_name_map = self.channel_name_map
    cdef PyObject *chnp = PyDict_GetItemWithError(channel_name_map, name)
    if chnp != NULL:
        return <int><object>chnp
    path = translate_channel(self.config, name)
    channel_path_map = self.channel_path_map
    chnp = PyDict_GetItemWithError(channel_path_map, path)
    if chnp != NULL:
        return <int><object>chnp
    channel_paths = self.channel_paths
    cid = PyList_GET_SIZE(channel_paths)
    assume_not_none(channel_paths)
    channel_paths.append(path)
    assume_not_none(channel_path_map)
    cdef object pycid = cid
    channel_path_map[path] = pycid
    assume_not_none(channel_name_map)
    channel_name_map[name] = pycid
    return cid

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
        seqinfo.C = new_param_pack(ParamPack, {}, {}, 'root', None)
        seqinfo.action_counter = 0
        self.seqinfo = seqinfo
        self.end_time = seqinfo.time_mgr.new_time_int(None, 0, False, True, None)
        self.seqinfo.bt_tracker.record(event_time_key(<void*>self.end_time))

    def __str__(self):
        io = StringIO()
        seq_show(self, io.write, 0)
        return io.getvalue()

    def __repr__(self):
        return str(self)

    cdef int finalize(self) except -1:
        seq_finalize(self, None, None, None)

    cdef int runtime_finalize(self, unsigned age, py_object &pyage) except -1:
        seq_runtime_finalize(self, age, pyage, None, None)

cdef int seq_show(Seq self, write, int indent) except -1:
    write(' ' * indent)
    write(f'Seq - T[{self.end_time.data.id}]\n')
    cdef int i = 0
    for t in self.seqinfo.time_mgr.event_times:
        write(' ' * (indent + 1))
        write(f'T[{i}]: ')
        write(str(t))
        write('\n')
        i += 1
    return subseq_show_subseqs(self, write, indent + 2)
