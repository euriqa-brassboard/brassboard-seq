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
from brassboard_seq.backend cimport CompiledSeq
from brassboard_seq.rtval cimport is_rtval

cimport cython

from brassboard_seq._utils import (JaqalInst_v1, Jaqal_v1, JaqalChannelGen_v1,
                                   JaqalInst_v1_3, Jaqal_v1_3)

cdef extern from "src/_rfsoc_backend.cpp" namespace "brassboard_seq::rfsoc_backend":
    void set_dds_delay(RFSOCBackend self, int dds, double delay) except +
    void rfsoc_finalize(RFSOCBackend ab, CompiledSeq&) except+
    void rfsoc_runtime_finalize(RFSOCBackend ab, CompiledSeq&, unsigned age) except +

    Generator *new_pulse_compiler_generator() except +
    cppclass PulseCompilerGen(Generator):
        object get_output()

    Generator *new_jaqal_pulse_compiler_generator() except +
    cppclass JaqalPulseCompilerGen(Generator):
        object get_prefix(int n) except +
        object get_sequence(int n) except +

    Generator *new_jaqalv1_3_stream_generator() except +
    cppclass Jaqalv1_3StreamGen(Generator):
        object get_prefix(int n) except +
        object get_sequence(int n) except +

cdef class RFSOCGenerator:
    pass

@cython.auto_pickle(False)
@cython.final
cdef class PulseCompilerGenerator(RFSOCGenerator):
    def __cinit__(self):
        self.gen.reset(new_pulse_compiler_generator())

    @property
    def output(self):
        return <dict>(<PulseCompilerGen*>self.gen.get()).get_output()

@cython.auto_pickle(False)
@cython.final
cdef class Jaqalv1Generator(RFSOCGenerator):
    def __cinit__(self):
        self.gen.reset(new_jaqal_pulse_compiler_generator())

    def get_prefix(self, n):
        return (<JaqalPulseCompilerGen*>self.gen.get()).get_prefix(n)

    def get_sequence(self, n):
        return (<JaqalPulseCompilerGen*>self.gen.get()).get_sequence(n)

@cython.auto_pickle(False)
@cython.final
cdef class Jaqalv1_3Generator(RFSOCGenerator):
    def __cinit__(self):
        self.gen.reset(new_jaqalv1_3_stream_generator())

    def get_prefix(self, n):
        return (<Jaqalv1_3StreamGen*>self.gen.get()).get_prefix(n)

    def get_sequence(self, n):
        return (<Jaqalv1_3StreamGen*>self.gen.get()).get_sequence(n)

@cython.auto_pickle(False)
@cython.final
cdef class RFSOCBackend:
    def __init__(self, RFSOCGenerator generator, /):
        self.eval_status = False
        self.generator = generator
        self.rt_dds_delay = {}
        self.use_all_channels = False

    @property
    def has_output(self):
        return self.channels.channels.size() != 0

    def set_dds_delay(self, int dds, delay, /):
        if is_rtval(delay):
            self.rt_dds_delay[dds] = delay
            return
        set_dds_delay(self, dds, <double>delay)

    cdef int finalize(self, CompiledSeq &cseq) except -1:
        rfsoc_finalize(self, cseq)

    cdef int runtime_finalize(self, CompiledSeq &cseq, unsigned age) except -1:
        rfsoc_runtime_finalize(self, cseq, age)
