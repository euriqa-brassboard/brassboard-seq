# cython: language_level=3

# Do not use relative import since it messes up cython file name tracking
from brassboard_seq.utils cimport PyErr_Format, PyExc_ValueError

cimport cython
from cpython cimport PyObject

@cython.auto_pickle(False)
cdef class Backend:
    cdef int finalize(self) except -1:
        pass

    cdef int runtime_finalize(self, unsigned age) except -1:
        pass

@cython.auto_pickle(False)
cdef class SeqCompiler:
    def __init__(self, seq, /):
        self.seq = seq
        self.backends = {}

    def add_backend(self, str name, Backend backend, /):
        if name in self.backends:
            PyErr_Format(PyExc_ValueError, 'Backend %U already exist', <PyObject*>name)
        self.backends[name] = backend
        backend.seq = self.seq
        backend.prefix = name

    def finalize(self, /):
        for _path in self.seq.seqinfo.channel_paths:
            path = <tuple>_path
            if path[0] not in self.backends:
                name = '/'.join(path)
                PyErr_Format(PyExc_ValueError, 'Unhandled channel: %U', <PyObject*>name)
        self.seq.finalize()
        for backend in self.backends.values():
            (<Backend>backend).finalize()

    def runtime_finalize(self, unsigned age, /):
        self.seq.runtime_finalize(age)
        for backend in self.backends.values():
            (<Backend>backend).runtime_finalize(age)
