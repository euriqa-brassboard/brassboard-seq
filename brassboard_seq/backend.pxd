# cython: language_level=3

# Do not use relative import since it messes up cython file name tracking
from brassboard_seq.seq cimport Seq

cdef class Backend:
    cdef Seq seq
    cdef str prefix

    cdef int finalize(self) except -1
    cdef int runtime_finalize(self, unsigned age) except -1

cdef class SeqCompiler:
    cdef readonly Seq seq
    cdef dict backends
