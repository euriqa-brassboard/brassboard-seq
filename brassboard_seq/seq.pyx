# cython: language_level=3

cimport cython
from cpython cimport PyErr_Format, PyObject

cdef class TimeSeq:
    def __init__(self):
        PyErr_Format(TypeError, "TimeSeq cannot be created directly")

@cython.no_gc
@cython.final
cdef class TimeStep(TimeSeq):
    def __init__(self):
        PyErr_Format(TypeError, "TimeStep cannot be created directly")

cdef class SubSeq(TimeSeq):
    def __init__(self):
        PyErr_Format(TypeError, "SubSeq cannot be created directly")

@cython.no_gc
@cython.final
cdef class SeqInfo:
    def __init__(self):
        PyErr_Format(TypeError, "SeqInfo cannot be created directly")

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
    self.seqinfo = seqinfo
    self.end_time = seqinfo.time_mgr.new_time_int(None, 0, False, True, None)
    return self
