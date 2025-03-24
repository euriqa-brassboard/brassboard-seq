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
from brassboard_seq.action cimport _RampFunctionBase
from brassboard_seq.seq cimport TimeStep
from brassboard_seq.utils cimport PyErr_Format, PyExc_ValueError

cimport cython
from cpython cimport PyObject, PyTypeObject

cdef extern from "src/backend.cpp" namespace "brassboard_seq::backend":
    PyTypeObject *timestep_type
    PyTypeObject *rampfunctionbase_type
    void compiler_finalize(SeqCompiler, TimeStep, _RampFunctionBase, Backend) except +
    void compiler_runtime_finalize(SeqCompiler, object age,
                                   _RampFunctionBase, Backend) except +

timestep_type = <PyTypeObject*>TimeStep
rampfunctionbase_type = <PyTypeObject*>_RampFunctionBase

@cython.auto_pickle(False)
cdef class Backend:
    cdef int finalize(self, CompiledSeq &cseq) except -1:
        pass

    cdef int runtime_finalize(self, CompiledSeq &cseq,
                              unsigned age, py_object &pyage) except -1:
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
        compiler_finalize(self, None, None, None)

    def runtime_finalize(self, age, /):
        compiler_runtime_finalize(self, age, None, None)
