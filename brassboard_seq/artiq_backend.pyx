# cython: language_level=3

# Copyright (c) 2024 - 2024 Yichao Yu <yyc1992@gmail.com>

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
from brassboard_seq.action cimport Action, RampFunction
from brassboard_seq.event_time cimport EventTime, round_time_int
from brassboard_seq.rtval cimport ExternCallback, is_rtval, new_extern, \
  RuntimeValue, rt_eval
from brassboard_seq.utils cimport set_global_tracker, PyErr_Format, \
  PyExc_RuntimeError, PyExc_TypeError, PyExc_ValueError, pyobject_call

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
        int SPI_CONFIG_ADDR
        int SPI_DATA_ADDR

    ArtiqConsts artiq_consts

    struct CompileVTable:
        bint (*is_rtval)(object) noexcept
        bint (*is_ramp)(object) noexcept

    void collect_actions(ArtiqBackend ab,
                         CompileVTable vtable, Action, EventTime) except +

    struct RuntimeVTable:
        object (*rt_eval)(object, unsigned)

    void generate_rtios(ArtiqBackend ab, unsigned age, RuntimeVTable vtable) except +

cdef inline bint is_ramp(obj) noexcept:
    return isinstance(obj, RampFunction)

cdef inline CompileVTable get_compile_vtable() noexcept nogil:
    cdef CompileVTable vt
    vt.is_rtval = is_rtval
    vt.is_ramp = is_ramp
    return vt

cdef inline RuntimeVTable get_runtime_vtable() noexcept nogil:
    cdef RuntimeVTable vt
    vt.rt_eval = <object (*)(object, unsigned)>rt_eval
    return vt

artiq_consts.COUNTER_ENABLE = <int?>edge_counter.CONFIG_COUNT_RISING | <int?>edge_counter.CONFIG_RESET_TO_ZERO
artiq_consts.COUNTER_DISABLE = <int?>edge_counter.CONFIG_SEND_COUNT_EVENT
artiq_consts._AD9910_REG_PROFILE0 = <int?>ad9910._AD9910_REG_PROFILE0
artiq_consts.URUKUL_CONFIG = <int?>urukul.SPI_CONFIG
artiq_consts.URUKUL_CONFIG_END = <int?>urukul.SPI_CONFIG | <int?>spi2.SPI_END
artiq_consts.URUKUL_SPIT_DDS_WR = <int?>urukul.SPIT_DDS_WR
artiq_consts.SPI_DATA_ADDR = <int?>spi2.SPI_DATA_ADDR
artiq_consts.SPI_CONFIG_ADDR = <int?>spi2.SPI_CONFIG_ADDR

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
    for _path in seq.seqinfo.channel_paths:
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
    def __init__(self, sys, cnpy.ndarray rtio_array, /):
        self.sys = sys
        self.eval_status = False
        if rtio_array.ndim != 1:
            PyErr_Format(PyExc_ValueError, "RTIO output must be a 1D array")
        if cnpy.PyArray_TYPE(rtio_array) != cnpy.NPY_INT32:
            PyErr_Format(PyExc_TypeError, "RTIO output must be a int32 array")
        self.rtio_array = rtio_array
        self.device_delay = {}

    cdef int add_start_trigger_ttl(self, uint32_t tgt, long long time,
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

    cdef int finalize(self) except -1:
        bt_guard = set_global_tracker(&self.seq.seqinfo.bt_tracker)
        collect_channels(&self.channels, self.prefix, self.sys, self.seq,
                         self.device_delay)
        collect_actions(self, get_compile_vtable(), None, None)

    cdef int runtime_finalize(self, unsigned age) except -1:
        bt_guard = set_global_tracker(&self.seq.seqinfo.bt_tracker)
        generate_rtios(self, age, get_runtime_vtable())

@cython.internal
@cython.auto_pickle(False)
@cython.final
cdef class EvalOnceCallback(ExternCallback):
    cdef object value
    cdef object callback

    def __call__(self):
        if self.value is None:
            PyErr_Format(PyExc_RuntimeError, 'Value evaluated too early')
        return self.value

    def __str__(self):
        return f'({self.callback})()'

@cython.internal
@cython.auto_pickle(False)
@cython.final
cdef class DatasetCallback(ExternCallback):
    cdef object value
    cdef object cb
    cdef tuple args
    cdef dict kwargs

    def __call__(self):
        if self.value is None:
            PyErr_Format(PyExc_RuntimeError, 'Value evaluated too early')
        return self.value

    def __str__(self):
        if self.args:
            name = self.args[0]
        else:
            name = '<unknown>'
        cb = self.cb
        if not PyMethod_Check(cb):
            return f'<dataset {name} for {self.cb}>'
        func = PyMethod_GET_FUNCTION(cb)
        obj = PyMethod_GET_SELF(cb)
        if <str?>(<object>func).__name__ == 'get_dataset_sys':
            return f'<dataset_sys {name} for {<object>obj}>'
        return f'<dataset {name} for {<object>obj}>'

cdef _eval_all_rtvals
def _eval_all_rtvals(self, /):
    try:
        vals = self._bb_rt_values
    except AttributeError:
        vals = ()
    if vals is None:
        return
    for val in vals:
        if type(val) is DatasetCallback:
            dval = <DatasetCallback>val
            dval.value = pyobject_call(dval.cb, dval.args, dval.kwargs)
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
        vals = []
        self._bb_rt_values = vals
    return vals

cdef rt_value
def rt_value(self, cb, /):
    vals = check_bb_rt_values(self)
    if vals is None:
        return cb()
    rtcb = <EvalOnceCallback>EvalOnceCallback.__new__(EvalOnceCallback)
    rtcb.callback = cb
    (<list?>vals).append(rtcb)
    return new_extern(rtcb)

cdef rt_dataset
def rt_dataset(self, /, *args, **kwargs):
    vals = check_bb_rt_values(self)
    cb = self.get_dataset
    if vals is None:
        return pyobject_call(cb, args, kwargs)
    rtcb = <DatasetCallback>DatasetCallback.__new__(DatasetCallback)
    rtcb.cb = cb
    rtcb.args = args
    rtcb.kwargs = kwargs
    (<list?>vals).append(rtcb)
    return new_extern(rtcb)

cdef rt_dataset_sys
def rt_dataset_sys(self, /, *args, **kwargs):
    vals = check_bb_rt_values(self)
    cb = self.get_dataset_sys
    if vals is None:
        return pyobject_call(cb, args, kwargs)
    rtcb = <DatasetCallback>DatasetCallback.__new__(DatasetCallback)
    rtcb.cb = cb
    rtcb.args = args
    rtcb.kwargs = kwargs
    (<list?>vals).append(rtcb)
    return new_extern(rtcb)

HasEnvironment.rt_value = rt_value
HasEnvironment.rt_dataset = rt_dataset
HasEnvironment.rt_dataset_sys = rt_dataset_sys
HasEnvironment._eval_all_rtvals = _eval_all_rtvals
