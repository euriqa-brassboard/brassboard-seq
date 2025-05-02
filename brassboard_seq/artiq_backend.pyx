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
from brassboard_seq.event_time cimport round_time_int
from brassboard_seq.rtval cimport ExternCallback, TagVal, is_rtval, new_extern
from brassboard_seq.utils cimport PyErr_Format, \
  PyExc_RuntimeError, PyExc_TypeError, PyExc_ValueError

from cpython cimport PyMethod_Check, PyMethod_GET_FUNCTION, PyMethod_GET_SELF

cimport cython
cimport numpy as cnpy
cnpy._import_array()

# Declare these as cdef so that they are hidden from python
# and can be accessed more efficiently from this module.
cdef artiq
import artiq.language.environment
cdef HasEnvironment = artiq.language.environment.HasEnvironment
cdef NoDefault = artiq.language.environment.NoDefault

cdef extern from "src/artiq_backend.cpp" namespace "brassboard_seq::artiq_backend":
    void artiq_add_start_trigger_ttl(ArtiqBackend ab, uint32_t tgt, int64_t time,
                                     int min_time, bint raising_edge) except +
    void artiq_add_start_trigger(ArtiqBackend ab, name, time, min_time, raising_edge) except +
    void artiq_finalize(ArtiqBackend ab, CompiledSeq&) except +
    void artiq_runtime_finalize(ArtiqBackend ab, CompiledSeq&, unsigned age) except +
    TagVal evalonce_callback(object) except +

@cython.auto_pickle(False)
@cython.final
cdef class ArtiqBackend:
    def __init__(self, sys, object rtio_array, /, *,
                 str output_format='bytecode'):
        self.sys = sys
        self.eval_status = False
        self.rtio_array = rtio_array
        self.device_delay = {}
        if output_format == 'bytecode':
            _rtio_array = <cnpy.ndarray?>rtio_array
            if _rtio_array.ndim != 1:
                PyErr_Format(PyExc_ValueError, "RTIO output must be a 1D array")
            if cnpy.PyArray_TYPE(_rtio_array) != cnpy.NPY_INT32:
                PyErr_Format(PyExc_TypeError, "RTIO output must be a int32 array")
            self.use_dma = False
        elif output_format == 'dma':
            <bytearray?>rtio_array
            self.use_dma = True
        else:
            PyErr_Format(PyExc_ValueError, "Unknown output type: '%U'",
                         <PyObject*>output_format)

    cdef int add_start_trigger_ttl(self, uint32_t tgt, int64_t time,
                                   int min_time, bint raising_edge) except -1:
        artiq_add_start_trigger_ttl(self, tgt, time, min_time, raising_edge)

    def add_start_trigger(self, str name, time, min_time, raising_edge, /):
        artiq_add_start_trigger(self, name, time, min_time, raising_edge)

    def set_device_delay(self, str name, delay, /):
        if is_rtval(delay):
            self.device_delay[name] = delay
            return
        if delay < 0:
            PyErr_Format(PyExc_ValueError, "Device time offset %S cannot be negative.",
                         <PyObject*>delay)
        if delay > 0.1:
            PyErr_Format(PyExc_ValueError, "Device time offset %S cannot be more than 100ms.",
                         <PyObject*>delay)
        self.device_delay[name] = round_time_int(delay)

    cdef int finalize(self, CompiledSeq &cseq) except -1:
        artiq_finalize(self, cseq)

    cdef int runtime_finalize(self, CompiledSeq &cseq, unsigned age) except -1:
        artiq_runtime_finalize(self, cseq, age)

@cython.internal
@cython.auto_pickle(False)
@cython.final
cdef class EvalOnceCallback(ExternCallback):
    cdef object value
    cdef object callback

    def __str__(self):
        return f'({self.callback})()'

@cython.internal
@cython.auto_pickle(False)
@cython.final
cdef class DatasetCallback(ExternCallback):
    cdef object value
    cdef object cb
    cdef str key
    cdef object default

    def __str__(self):
        cb = self.cb
        if not PyMethod_Check(cb):
            return f'<dataset {self.key} for {self.cb}>'
        func = PyMethod_GET_FUNCTION(cb)
        obj = PyMethod_GET_SELF(cb)
        if <str?>(<object>func).__name__ == 'get_dataset_sys':
            return f'<dataset_sys {self.key} for {<object>obj}>'
        return f'<dataset {self.key} for {<object>obj}>'

cdef _eval_all_rtvals
cdef dict _empty_dict = {}
def _eval_all_rtvals(self, /):
    try:
        vals = self._bb_rt_values
    except AttributeError:
        vals = _empty_dict
    if vals is None:
        return
    for val in (<dict?>vals).values():
        if type(val) is DatasetCallback:
            dval = <DatasetCallback>val
            dval.value = dval.cb(dval.key, dval.default)
        elif type(val) is EvalOnceCallback:
            eoval = <EvalOnceCallback>val
            eoval.value = eoval.callback()
        else:
            PyErr_Format(PyExc_RuntimeError, 'Unknown object in runtime callbacks')
    self._bb_rt_values = None
    self.call_child_method('_eval_all_rtvals')

cdef inline check_bb_rt_values(self):
    try:
        vals = self._bb_rt_values # may be None
    except AttributeError:
        vals = {}
        self._bb_rt_values = vals
    return vals

cdef rt_value
def rt_value(self, cb, /):
    vals = check_bb_rt_values(self)
    if vals is None:
        return cb()
    rtcb = <EvalOnceCallback>EvalOnceCallback.__new__(EvalOnceCallback)
    rtcb.fptr = <void*><TagVal(*)(EvalOnceCallback)>evalonce_callback
    rtcb.callback = cb
    (<dict?>vals)[cb] = rtcb
    return new_extern(rtcb, float)

cdef rt_dataset
def rt_dataset(self, str key, default=NoDefault):
    _vals = check_bb_rt_values(self)
    cb = self.get_dataset
    if _vals is None:
        return cb(key, default)
    vals = <dict?>_vals
    res = vals.get((key, False))
    if res is not None:
        return new_extern(res, float)
    rtcb = <DatasetCallback>DatasetCallback.__new__(DatasetCallback)
    rtcb.fptr = <void*><TagVal(*)(DatasetCallback)>evalonce_callback
    rtcb.cb = cb
    rtcb.key = key
    rtcb.default = default
    vals[(key, False)] = rtcb
    return new_extern(rtcb, float)

cdef rt_dataset_sys
def rt_dataset_sys(self, str key, default=NoDefault):
    _vals = check_bb_rt_values(self)
    cb = self.get_dataset_sys
    if _vals is None:
        return cb(key, default)
    vals = <dict?>_vals
    res = vals.get((key, True))
    if res is not None:
        return new_extern(res, float)
    rtcb = <DatasetCallback>DatasetCallback.__new__(DatasetCallback)
    rtcb.fptr = <void*><TagVal(*)(DatasetCallback)>evalonce_callback
    rtcb.cb = cb
    rtcb.key = key
    rtcb.default = default
    vals[(key, True)] = rtcb
    return new_extern(rtcb, float)

HasEnvironment.rt_value = rt_value
HasEnvironment.rt_dataset = rt_dataset
HasEnvironment.rt_dataset_sys = rt_dataset_sys
HasEnvironment._eval_all_rtvals = _eval_all_rtvals
