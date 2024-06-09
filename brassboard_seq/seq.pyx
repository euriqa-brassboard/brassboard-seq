# cython: language_level=3

# Do not use relative import since it messes up cython file name tracking
from brassboard_seq.config cimport translate_channel
from brassboard_seq.rtval cimport convert_bool
from brassboard_seq.utils cimport assume_not_none, _assume_not_none, \
  event_time_key

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
        return _get_channel_id(self.seqinfo, name)

@cython.no_gc
@cython.final
cdef class TimeStep(TimeSeq):
    def __init__(self):
        PyErr_Format(TypeError, "TimeStep cannot be created directly")


@cython.final
cdef class ConditionalWrapper:
    def __init__(self):
        PyErr_Format(TypeError, "ConditionalWrapper cannot be created directly")

    def conditional(self, cond):
        wrapper = <ConditionalWrapper>ConditionalWrapper.__new__(ConditionalWrapper)
        wrapper.seq = self.seq
        wrapper.cond = combine_cond(self.cond, cond)
        return wrapper

    def wait(self, length, *, cond=True):
        wait_cond(self.seq, length, combine_cond(self.cond, cond))

    def add_step(self, first_arg, *args, **kwargs):
        seq = self.seq
        step = add_step_real(seq, self.cond, seq.end_time, first_arg, args, kwargs)
        seq.end_time = step.end_time
        return step

    def add_background(self, first_arg, *args, **kwargs):
        seq = self.seq
        return add_step_real(seq, self.cond, seq.end_time, first_arg, args, kwargs)

    def add_floating(self, first_arg, *args, **kwargs):
        seq = self.seq
        cond = self.cond
        return add_step_real(seq, cond,
                             seq.seqinfo.time_mgr.new_time_int(None, 0, True,
                                                               cond, None),
                             first_arg, args, kwargs)

    def add_at(self, EventTime tp, first_arg, *args, **kwargs):
        seq = self.seq
        return add_step_real(seq, self.cond, tp, first_arg, args, kwargs)

cdef class SubSeq(TimeSeq):
    def __init__(self):
        PyErr_Format(TypeError, "SubSeq cannot be created directly")

    def conditional(self, cond):
        wrapper = <ConditionalWrapper>ConditionalWrapper.__new__(ConditionalWrapper)
        wrapper.seq = self
        wrapper.cond = combine_cond(self.cond, cond)
        return wrapper

    def wait(self, length, *, cond=True):
        wait_cond(self, length, combine_cond(self.cond, cond))

    def add_step(self, first_arg, *args, **kwargs):
        step = add_step_real(self, self.cond, self.end_time, first_arg, args, kwargs)
        self.end_time = step.end_time
        return step

    def add_background(self, first_arg, *args, **kwargs):
        return add_step_real(self, self.cond, self.end_time, first_arg, args, kwargs)

    def add_floating(self, first_arg, *args, **kwargs):
        cond = self.cond
        return add_step_real(self, cond,
                             self.seqinfo.time_mgr.new_time_int(None, 0, True,
                                                                cond, None),
                             first_arg, args, kwargs)

    def add_at(self, EventTime tp, first_arg, *args, **kwargs):
        return add_step_real(self, self.cond, tp, first_arg, args, kwargs)

cdef int wait_cond(SubSeq self, length, cond) except -1:
    self.end_time = self.seqinfo.time_mgr.new_round_time(self.end_time, length,
                                                         False, cond, None)
    self.seqinfo.bt_tracker.record(event_time_key(<void*>self.end_time))
    return 0

cdef SubSeq add_custom_step(SubSeq self, cond, EventTime start_time, cb,
                            tuple args, dict kwargs):
    subseq = <SubSeq>SubSeq.__new__(SubSeq)
    init_subseq(subseq, self, start_time, cond)
    cb(subseq, *args, **kwargs)
    _assume_not_none(<void*>self.sub_seqs)
    self.sub_seqs.append(subseq)
    return subseq

cdef TimeStep add_time_step(SubSeq self, cond, EventTime start_time, length):
    step = <TimeStep>TimeStep.__new__(TimeStep)
    init_timeseq(step, self, start_time, cond)
    step.actions = {}
    step.length = length
    step.end_time = self.seqinfo.time_mgr.new_round_time(start_time, length,
                                                         False, cond, None)
    self.seqinfo.bt_tracker.record(event_time_key(<void*>step.end_time))
    _assume_not_none(<void*>self.sub_seqs)
    self.sub_seqs.append(step)
    return step

cdef TimeSeq add_step_real(SubSeq self, cond, EventTime start_time,
                           first_arg, tuple args, dict kwargs):
    if callable(first_arg):
        return add_custom_step(self, cond, start_time, first_arg, args, kwargs)
    elif not PyTuple_GET_SIZE(args) and not PyDict_Size(kwargs):
        return add_time_step(self, cond, start_time, first_arg)
    else:
        sargs = str(args)
        skwargs = str(kwargs)
        PyErr_Format(ValueError,
                     "Unexpected arguments when creating new time step, %U, %U.",
                     <PyObject*>sargs, <PyObject*>skwargs)

@cython.no_gc
@cython.final
cdef class SeqInfo:
    def __init__(self):
        PyErr_Format(TypeError, "SeqInfo cannot be created directly")

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

@cython.final
cdef class Seq(SubSeq):
    def __init__(self, Config config, int max_frame=0):
        init_subseq(self, None, None, True)
        seqinfo = <SeqInfo>SeqInfo.__new__(SeqInfo)
        seqinfo.config = config
        seqinfo.time_mgr = new_time_manager()
        seqinfo.bt_tracker.max_frame = max_frame
        seqinfo.channel_name_map = {}
        seqinfo.channel_path_map = {}
        seqinfo.channel_paths = []
        self.seqinfo = seqinfo
        self.end_time = seqinfo.time_mgr.new_time_int(None, 0, False, True, None)
        self.seqinfo.bt_tracker.record(event_time_key(<void*>self.end_time))

cdef inline void init_timeseq(TimeSeq self, SubSeq parent,
                              EventTime start_time, cond) noexcept:
    if parent is not None:
        self.seqinfo = parent.seqinfo
    self.start_time = start_time
    self.cond = cond

cdef inline void init_subseq(SubSeq self, SubSeq parent, EventTime start_time, cond) noexcept:
    init_timeseq(self, parent, start_time, cond)
    self.end_time = start_time
    self.sub_seqs = []
