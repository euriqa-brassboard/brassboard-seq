# cython: language_level=3

# Do not use relative import since it messes up cython file name tracking
from brassboard_seq.rtval cimport convert_bool
from brassboard_seq.utils cimport assume_not_none

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
        return self.seqinfo._get_channel_id(name)

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

cdef class SubSeq(TimeSeq):
    def __init__(self):
        PyErr_Format(TypeError, "SubSeq cannot be created directly")

    cpdef ConditionalWrapper conditional(self, cond):
        wrapper = <ConditionalWrapper>ConditionalWrapper.__new__(ConditionalWrapper)
        wrapper.seq = self
        wrapper.cond = combine_cond(self.cond, cond)
        return wrapper

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
