# cython: language_level=3

# Do not use relative import since it messes up cython file name tracking
from brassboard_seq.config cimport translate_channel
from brassboard_seq.rtval cimport convert_bool
from brassboard_seq.utils cimport assume_not_none, event_time_key

cimport cython
from cpython cimport PyErr_Format, PyObject, PyDict_GetItemWithError, PyList_GET_SIZE

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

cdef int wait_cond(SubSeq self, length, cond) except -1:
    self.end_time = self.seqinfo.time_mgr.new_round_time(self.end_time, length,
                                                         False, cond, None)
    self.seqinfo.bt_tracker.record(event_time_key(<void*>self.end_time))
    return 0

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
