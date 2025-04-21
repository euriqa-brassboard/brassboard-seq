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
from brassboard_seq.backend cimport CompiledSeq
from brassboard_seq.event_time cimport round_time_int
from brassboard_seq.rtval cimport ExternCallback, TagVal, is_rtval, new_extern
from brassboard_seq.seq cimport Seq, seq_get_channel_paths
from brassboard_seq.utils cimport set_global_tracker, PyErr_Format, \
  PyExc_RuntimeError, PyExc_TypeError, PyExc_ValueError

from cpython cimport PyMethod_Check, PyMethod_GET_FUNCTION, PyMethod_GET_SELF

cimport cython
cimport numpy as cnpy
cnpy._import_array()

# Declare these as cdef so that they are hidden from python
# and can be accessed more efficiently from this module.
cdef artiq, ad9910, edge_counter, spi2, ttl, urukul

import artiq.language.environment
from artiq.coredevice import ad9910, edge_counter, spi2, ttl, urukul

cdef HasEnvironment = artiq.language.environment.HasEnvironment
cdef NoDefault = artiq.language.environment.NoDefault
cdef DevAD9910 = ad9910.AD9910
cdef DevEdgeCounter = edge_counter.EdgeCounter
cdef DevTTLOut = ttl.TTLOut

cdef sim_ad9910, sim_edge_counter, sim_ttl
try:
    from dax.sim.coredevice import (ad9910 as sim_ad9910,
                                    edge_counter as sim_edge_counter,
                                    ttl as sim_ttl)
    DevAD9910 = (DevAD9910, sim_ad9910.AD9910)
    DevEdgeCounter = (DevEdgeCounter, sim_edge_counter.EdgeCounter)
    DevTTLOut = (DevTTLOut, sim_ttl.TTLOut)
except:
    pass

cdef extern from "src/artiq_backend.cpp" namespace "brassboard_seq::artiq_backend":
    struct ArtiqConsts:
        int COUNTER_ENABLE
        int COUNTER_DISABLE
        int _AD9910_REG_PROFILE0
        int URUKUL_CONFIG
        int URUKUL_CONFIG_END
        int URUKUL_SPIT_DDS_WR
        int URUKUL_DEFAULT_PROFILE
        int SPI_CONFIG_ADDR
        int SPI_DATA_ADDR

    ArtiqConsts artiq_consts
    PyObject *rampfunctionbase_type

    void collect_actions(ArtiqBackend ab, CompiledSeq&) except +

    void generate_rtios(ArtiqBackend ab, CompiledSeq&, unsigned age) except +
    TagVal evalonce_callback(object) except +

artiq_consts.COUNTER_ENABLE = <int?>edge_counter.CONFIG_COUNT_RISING | <int?>edge_counter.CONFIG_RESET_TO_ZERO
artiq_consts.COUNTER_DISABLE = <int?>edge_counter.CONFIG_SEND_COUNT_EVENT
artiq_consts._AD9910_REG_PROFILE0 = <int?>ad9910._AD9910_REG_PROFILE0
artiq_consts.URUKUL_CONFIG = <int?>urukul.SPI_CONFIG
artiq_consts.URUKUL_CONFIG_END = <int?>urukul.SPI_CONFIG | <int?>spi2.SPI_END
artiq_consts.URUKUL_SPIT_DDS_WR = <int?>urukul.SPIT_DDS_WR
artiq_consts.URUKUL_DEFAULT_PROFILE = urukul.DEFAULT_PROFILE if hasattr(urukul, 'DEFAULT_PROFILE') else 0
artiq_consts.SPI_DATA_ADDR = <int?>spi2.SPI_DATA_ADDR
artiq_consts.SPI_CONFIG_ADDR = <int?>spi2.SPI_CONFIG_ADDR
rampfunctionbase_type = <PyObject*>_RampFunctionBase

cdef PyObject *raise_invalid_channel(tuple path) except NULL:
    name = '/'.join(path)
    return PyErr_Format(PyExc_ValueError, 'Invalid channel name %U', <PyObject*>name)

cdef get_artiq_device(sys, str name):
    if hasattr(sys, 'registry'):
        # DAX support
        unique = <str?>sys.registry.get_unique_device_key(name)
    else:
        unique = name
    # Do not call the get_device function from DAX since
    # it assumes that the calling object will take ownership of the deivce.
    return HasEnvironment.get_device(sys, unique)

cdef int add_channel_artiq(ChannelsInfo *self, dev, int64_t delay, PyObject *rt_delay,
                           int idx, tuple path) except -1:
    cdef ChannelType dds_param_type
    if isinstance(dev, DevAD9910):
        if len(path) != 3:
            raise_invalid_channel(path)
        path2 = <str>path[2]
        if path2 == 'sw':
            # Note that we currently do not treat this switch ttl channel
            # differently from any other ttl channels.
            # We may consider maintaining a relation between this ttl channel
            # and the urukul channel to make sure we don't reorder
            # any operations between the two.
            self.add_ttl_channel(idx, <int?>dev.sw.target_o, False, delay, rt_delay)
            return 0
        elif path2 == 'freq':
            dds_param_type = DDSFreq
        elif path2 == 'amp':
            dds_param_type = DDSAmp
        elif path2 == 'phase':
            dds_param_type = DDSPhase
        else:
            # Make the C compiler happy since it doesn't know
            # that `raise_invalid_channel` doesn't return
            dds_param_type = DDSPhase
            raise_invalid_channel(path)
        bus = dev.bus
        bus_channel = <int?>bus.channel
        bus_id = self.find_bus_id(bus_channel)
        if bus_id == -1:
            # Here we assume that the CPLD (and it's io_update channel)
            # and the SPI bus has a one-to-one mapping.
            # This means that each DDS with the same bus shares
            # the same io_update channel and can only be programmed one at a time.
            io_update_target = <int?>dev.cpld.io_update.target_o
            bus_id = self.add_bus_channel(bus_channel, io_update_target,
                                          <int?>bus.ref_period_mu)
        self.add_dds_param_channel(idx, bus_id, <double?>dev.ftw_per_hz,
                                   <int?>dev.chip_select, dds_param_type,
                                   delay, rt_delay)
    elif isinstance(dev, DevTTLOut):
        if len(path) > 2:
            raise_invalid_channel(path)
        self.add_ttl_channel(idx, <int?>dev.target_o, False, delay, rt_delay)
    elif isinstance(dev, DevEdgeCounter):
        if len(path) > 2:
            raise_invalid_channel(path)
        self.add_ttl_channel(idx, (<int?>dev.channel) << 8, True, delay, rt_delay)
    else:
        devstr = str(dev)
        PyErr_Format(PyExc_ValueError, 'Unsupported device: %U', <PyObject*>devstr)
    return 0

cdef int collect_channels(ChannelsInfo *self, str prefix, sys, Seq seq,
                          dict device_delay) except -1:
    cdef int idx = -1
    cdef int64_t delay
    cdef PyObject *rt_delay
    for _path in seq_get_channel_paths(seq):
        idx += 1
        path = <tuple>_path
        if <str>path[0] != prefix:
            continue
        if len(path) < 2:
            raise_invalid_channel(path)
        devname = <str>path[1]
        py_delay = device_delay.get(devname)
        if py_delay is None:
            delay = 0
            rt_delay = NULL
        elif is_rtval(py_delay):
            delay = 0
            rt_delay = <PyObject*>py_delay
        else:
            delay = py_delay
            rt_delay = NULL
        add_channel_artiq(self, get_artiq_device(sys, devname), delay, rt_delay,
                          idx, path)
    self.dds_chn_map.clear() # Not needed after channel collection
    return 0

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
        cdef StartTrigger start_trigger
        start_trigger.target = tgt
        start_trigger.min_time_mu = <uint16_t>max(seq_time_to_mu(min_time), 8)
        start_trigger.time_mu = seq_time_to_mu(time)
        start_trigger.raising_edge = raising_edge
        self.start_triggers.push_back(start_trigger)
        return 0

    def add_start_trigger(self, str name, time, min_time,
                          bint raising_edge, /):
        dev = get_artiq_device(self.sys, name)
        if not isinstance(dev, DevTTLOut):
            PyErr_Format(PyExc_ValueError, 'Invalid start trigger device: %U',
                         <PyObject*>name)
        self.add_start_trigger_ttl(dev.target_o, round_time_int(time),
                                   round_time_int(min_time), raising_edge)

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
        collect_channels(&self.channels, self.prefix, self.sys, self.seq,
                         self.device_delay)
        collect_actions(self, cseq)

    cdef int runtime_finalize(self, CompiledSeq &cseq, unsigned age) except -1:
        generate_rtios(self, cseq, age)

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
