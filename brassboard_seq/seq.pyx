# cython: language_level=3

# Do not use relative import since it messes up cython file name tracking
from brassboard_seq.event_time cimport round_time_int, round_time_rt
from brassboard_seq.rtval cimport convert_bool, is_rtval, RuntimeValue

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

@cython.no_gc
@cython.final
cdef class TimeStep(TimeSeq):
    def __init__(self):
        PyErr_Format(TypeError, "TimeStep cannot be created directly")


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

    @property
    def current_time(self):
        return self.end_time

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
        channel_paths.append(path)
        cdef object pycid = cid
        channel_path_map[path] = pycid
        channel_name_map[name] = pycid
        return cid

@cython.final
cdef class Seq(SubSeq):
    def __init__(self):
        PyErr_Format(TypeError, "Seq cannot be created directly")

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
