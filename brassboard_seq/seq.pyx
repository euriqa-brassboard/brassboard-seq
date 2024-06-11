# cython: language_level=3

# Do not use relative import since it messes up cython file name tracking
from brassboard_seq.action cimport new_action
from brassboard_seq.event_time cimport round_time_int, round_time_rt
from brassboard_seq.rtval cimport convert_bool, is_rtval, RuntimeValue
from brassboard_seq.utils cimport assume_not_none, _assume_not_none

import io
cdef StringIO = io.StringIO

cimport cython
from cpython cimport PyErr_Format, PyObject, PyDict_GetItemWithError, PyList_GET_SIZE, PyTuple_GET_SIZE, PyDict_Size

cdef combine_cond(cond1, new_cond):
    if cond1 is False:
        return False
    cond2 = convert_bool(new_cond)
    if cond1 is True:
        return cond2
    if cond2 is True:
        return cond1
    if cond2 is False:
        return False
    return cond1 & cond2

cdef class TimeSeq:
    def __init__(self):
        PyErr_Format(TypeError, "TimeSeq cannot be created directly")

    def get_channel_id(self, str name):
        return self.seqinfo._get_channel_id(name)

    def set_time(self, EventTime time, offset=0): # offset in seconds
        if is_rtval(offset):
            self.start_time.set_base_rt(time, round_time_rt(<RuntimeValue>offset))
        else:
            self.start_time.set_base_int(time, round_time_int(offset))

cdef int timestep_show(TimeStep self, write, int indent) except -1:
    write(' ' * indent)
    write(f'TimeStep({self.length})@T[{self.start_time.id}]')
    cond = self.cond
    if cond is not True:
        write(f' if {cond}\n')
    else:
        write('\n')
    for (chn, action) in self.actions.items():
        chn = '/'.join(self.seqinfo.channel_paths[chn])
        write(' ' * (indent + 2))
        write(f'{chn}: {str(action)}\n')
    return 0

@cython.no_gc
@cython.final
cdef class TimeStep(TimeSeq):
    def __init__(self):
        PyErr_Format(TypeError, "TimeStep cannot be created directly")

    cdef int _set(self, chn, value, cond, bint is_pulse, bint exact_time, dict kws) except -1:
        cdef int cid
        if type(chn) is int:
            cid = <int>chn
            pycid = chn
        else:
            cid = self.seqinfo._get_channel_id(chn)
            pycid = cid
        cdef dict actions = self.actions
        assume_not_none(actions)
        if pycid in actions:
            PyErr_Format(ValueError, "Multiple action added for the same channel.")
        action = new_action(value, cond, is_pulse, exact_time, kws)
        assume_not_none(actions)
        actions[pycid] = action
        return 0

    def set(self, chn, value, *, cond=True, bint exact_time=False, **kws):
        self._set(chn, value, combine_cond(self.cond, cond), False, exact_time, kws)
        return self

    def pulse(self, chn, value, *, cond=True, bint exact_time=False, **kws):
        self._set(chn, value, combine_cond(self.cond, cond), True, exact_time, kws)
        return self

    def __str__(self):
        io = StringIO()
        timestep_show(self, io.write, 0)
        return io.getvalue()

    def __repr__(self):
        return str(self)

cdef int conditionalwrapper_show(ConditionalWrapper self, write, int indent) except -1:
    write(' ' * indent)
    write(f'ConditionalWrapper({self.cond}) for\n')
    if type(self.seq) == Seq:
        return seq_show(self.seq, write, indent + 2)
    return subseq_show(self.seq, write, indent + 2)

@cython.final
cdef class ConditionalWrapper:
    def __init__(self):
        PyErr_Format(TypeError, "ConditionalWrapper cannot be created directly")

    cpdef ConditionalWrapper conditional(self, cond):
        wrapper = <ConditionalWrapper>ConditionalWrapper.__new__(ConditionalWrapper)
        wrapper.seq = self.seq
        wrapper.cond = combine_cond(self.cond, cond)
        return wrapper

    def wait(self, length, *, cond=True):
        self.seq.wait_cond(length, combine_cond(self.cond, cond))

    def add_step(self, first_arg, *args, **kwargs):
        seq = self.seq
        step = seq.add_step_real(self.cond, seq.end_time, first_arg, args, kwargs)
        seq.end_time = step.end_time
        return step

    def add_background(self, first_arg, *args, **kwargs):
        seq = self.seq
        return seq.add_step_real(self.cond, seq.end_time, first_arg, args, kwargs)

    def add_floating(self, first_arg, *args, **kwargs):
        seq = self.seq
        cond = self.cond
        return seq.add_step_real(cond,
                                 seq.seqinfo.time_mgr.new_time_int(None, 0, True,
                                                                   cond, None),
                                 first_arg, args, kwargs)

    def add_at(self, EventTime tp, first_arg, *args, **kwargs):
        seq = self.seq
        return seq.add_step_real(self.cond, tp, first_arg, args, kwargs)

    def wait_for(self, tp, offset=0):
        self.seq.wait_for_cond(tp, offset, self.cond)

    def set(self, chn, value, *, cond=True, bint exact_time=False, **kws):
        self.seq._set(chn, value, combine_cond(self.cond, cond), exact_time, kws)
        return self

    def __str__(self):
        io = StringIO()
        conditionalwrapper_show(self, io.write, 0)
        return io.getvalue()

    def __repr__(self):
        return str(self)

cdef int subseq_show_subseqs(SubSeq self, write, int indent) except -1:
    for _subseq in self.sub_seqs:
        subseq = <TimeSeq>_subseq
        if subseq.is_step:
            timestep_show(<TimeStep>subseq, write, indent)
        else:
            subseq_show(<SubSeq>subseq, write, indent)
    return 0

cdef int subseq_show(SubSeq self, write, int indent) except -1:
    write(' ' * indent)
    write(f'SubSeq@T[{self.start_time.id}] - T[{self.end_time.id}]')
    cond = self.cond
    if cond is not True:
        write(f' if {cond}\n')
    else:
        write('\n')
    subseq_show_subseqs(self, write, indent + 2)
    return 0

cdef class SubSeq(TimeSeq):
    def __init__(self):
        PyErr_Format(TypeError, "SubSeq cannot be created directly")

    cpdef ConditionalWrapper conditional(self, cond):
        wrapper = <ConditionalWrapper>ConditionalWrapper.__new__(ConditionalWrapper)
        wrapper.seq = self
        wrapper.cond = combine_cond(self.cond, cond)
        return wrapper

    @cython.final
    cdef int wait_cond(self, length, cond) except -1:
        self.end_time = self.seqinfo.time_mgr.new_round_time(self.end_time, length,
                                                            False, cond, None)
        return 0

    def wait(self, length, *, cond=True):
        self.wait_cond(length, combine_cond(self.cond, cond))

    @cython.final
    cdef SubSeq add_custom_step(self, cond, EventTime start_time, cb,
                                tuple args, dict kwargs):
        subseq = <SubSeq>SubSeq.__new__(SubSeq)
        init_subseq(subseq, self, start_time, cond)
        cb(subseq, *args, **kwargs)
        _assume_not_none(<void*>self.sub_seqs)
        self.sub_seqs.append(subseq)
        return subseq

    @cython.final
    cdef TimeStep add_time_step(self, cond, EventTime start_time, length):
        step = <TimeStep>TimeStep.__new__(TimeStep)
        init_timeseq(step, self, start_time, cond, True)
        step.actions = {}
        step.length = length
        step.end_time = self.seqinfo.time_mgr.new_round_time(start_time, length,
                                                             False, cond, None)
        _assume_not_none(<void*>self.sub_seqs)
        self.sub_seqs.append(step)
        return step

    @cython.final
    cdef TimeSeq add_step_real(self, cond, EventTime start_time,
                               first_arg, tuple args, dict kwargs):
        if callable(first_arg):
            return self.add_custom_step(cond, start_time, first_arg, args, kwargs)
        elif not PyTuple_GET_SIZE(args) and not PyDict_Size(kwargs):
            return self.add_time_step(cond, start_time, first_arg)
        else:
            sargs = str(args)
            skwargs = str(kwargs)
            PyErr_Format(ValueError,
                         "Unexpected arguments when creating new time step, %U, %U.",
                         <PyObject*>sargs, <PyObject*>skwargs)

    def add_step(self, first_arg, *args, **kwargs):
        step = self.add_step_real(self.cond, self.end_time, first_arg, args, kwargs)
        self.end_time = step.end_time
        return step

    def add_background(self, first_arg, *args, **kwargs):
        return self.add_step_real(self.cond, self.end_time, first_arg, args, kwargs)

    def add_floating(self, first_arg, *args, **kwargs):
        cond = self.cond
        return self.add_step_real(cond,
                                  self.seqinfo.time_mgr.new_time_int(None, 0, True,
                                                                     cond, None),
                                  first_arg, args, kwargs)

    def add_at(self, EventTime tp, first_arg, *args, **kwargs):
        return self.add_step_real(self.cond, tp, first_arg, args, kwargs)

    @cython.final
    cdef int wait_for_cond(self, _tp0, offset, cond) except -1:
        cdef EventTime tp0
        if type(_tp0) is EventTime:
            tp0 = <EventTime>_tp0
        else:
            tp0 = (<TimeSeq?>_tp0).end_time
        self.end_time = self.seqinfo.time_mgr.new_round_time(self.end_time, offset,
                                                             False, cond, tp0)
        return 0

    def wait_for(self, tp, offset=0):
        self.wait_for_cond(tp, offset, self.cond)

    @property
    def current_time(self):
        return self.end_time

    cdef int _set(self, chn, value, cond, bint exact_time, dict kws) except -1:
        step = self.dummy_step
        start_time = self.end_time
        if step is None or step.end_time is not start_time:
            step = self.add_time_step(self.cond, start_time, 0)
            self.dummy_step = step
            # Update the current time so that a normal step added later
            # this is treated as ordered after this set event
            # rather than at the same time.
            self.end_time = step.end_time
        return step._set(chn, value, cond, False, exact_time, kws)

    def set(self, chn, value, *, cond=True, bint exact_time=False, **kws):
        self._set(chn, value, combine_cond(self.cond, cond), exact_time, kws)
        return self

    def __str__(self):
        io = StringIO()
        subseq_show(self, io.write, 0)
        return io.getvalue()

    def __repr__(self):
        return str(self)

cdef int seq_show(Seq self, write, int indent) except -1:
    write(' ' * indent)
    write(f'Seq - T[{self.end_time.id}]\n')
    cdef int i = 0
    for t in self.seqinfo.time_mgr.event_times:
        write(' ' * (indent + 1))
        write(f'T[{i}]: ')
        write(str(t))
        write('\n')
        i += 1
    return subseq_show_subseqs(self, write, indent + 2)

@cython.no_gc
@cython.final
cdef class SeqInfo:
    def __init__(self):
        PyErr_Format(TypeError, "SeqInfo cannot be created directly")

    cdef int _get_channel_id(self, str name) except -1:
        channel_name_map = self.channel_name_map
        cdef PyObject *chnp = PyDict_GetItemWithError(channel_name_map, name)
        if chnp != NULL:
            return <int><object>chnp
        path = self.config.translate_channel(name)
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

@cython.final
cdef class Seq(SubSeq):
    def __init__(self):
        PyErr_Format(TypeError, "Seq cannot be created directly")

    def __str__(self):
        io = StringIO()
        seq_show(self, io.write, 0)
        return io.getvalue()

    def __repr__(self):
        return str(self)

cdef inline void init_timeseq(TimeSeq self, SubSeq parent,
                              EventTime start_time, cond, bint is_step) noexcept:
    if parent is not None:
        self.seqinfo = parent.seqinfo
    self.start_time = start_time
    self.cond = cond
    self.is_step = is_step

cdef inline void init_subseq(SubSeq self, SubSeq parent, EventTime start_time, cond) noexcept:
    init_timeseq(self, parent, start_time, cond, False)
    self.end_time = start_time
    self.sub_seqs = []

cpdef Seq new_seq(Config config):
    self = <Seq>Seq.__new__(Seq)
    init_subseq(self, None, None, True)
    seqinfo = <SeqInfo>SeqInfo.__new__(SeqInfo)
    seqinfo.config = config
    seqinfo.time_mgr = new_time_manager()
    seqinfo.channel_name_map = {}
    seqinfo.channel_path_map = {}
    seqinfo.channel_paths = []
    self.seqinfo = seqinfo
    self.end_time = seqinfo.time_mgr.new_time_int(None, 0, False, True, None)
    return self
