# cython: language_level=3

cimport cython
from cpython cimport PyErr_Format, PyObject

# Declare these as cdef so that they are hidden from python
# and can be accessed more efficiently from this module.
cdef artiq, ad9910, edge_counter, spi2, ttl, urukul

import artiq.language.environment
from artiq.coredevice import ad9910, edge_counter, spi2, ttl, urukul

cdef DevAD9910 = ad9910.AD9910
cdef DevEdgeCounter = edge_counter.EdgeCounter
cdef DevTTLOut = ttl.TTLOut

cdef extern from "../src/artiq_backend.cpp" namespace "artiq_backend" nogil:
    ArtiqConsts artiq_consts

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
    return PyErr_Format(ValueError, 'Invalid channel name %U', <PyObject*>name)

@cython.final
cdef class ChannelsInfo:
    def __init__(self):
        PyErr_Format(TypeError, "ChannelInfo cannot be created directly")

    cdef int add_bus_channel(self, int bus_channel, int io_update_target,
                             int ref_period_mu) except -1:
        cdef UrukulBus bus_info
        bus_info.channel = bus_channel
        bus_info.addr_target = (bus_channel << 8) | artiq_consts.SPI_CONFIG_ADDR
        bus_info.data_target = (bus_channel << 8) | artiq_consts.SPI_DATA_ADDR
        # Here we assume that the CPLD (and it's io_update channel)
        # and the SPI bus has a one-to-one mapping.
        # This means that each DDS with the same bus shares
        # the same io_update channel and can only be programmed one at a time.
        bus_info.io_update_target = io_update_target
        bus_info.ref_period_mu = ref_period_mu
        cdef int bus_id = self.urukul_busses.size()
        self.urukul_busses.push_back(bus_info)
        self.bus_chn_map[bus_channel] = bus_id
        return bus_id

    cdef int add_ttl_channel(self, int seqchn, int target, bint iscounter) except -1:
        assert self.ttl_chn_map.count(seqchn) == 0
        cdef TTLChannel ttl_chn
        ttl_chn.target = target
        ttl_chn.iscounter = iscounter
        cdef int ttl_id = self.ttlchns.size()
        self.ttlchns.push_back(ttl_chn)
        self.ttl_chn_map[seqchn] = ttl_id
        return ttl_id

    cdef int get_dds_channel_id(self, int bus_id, double ftw_per_hz,
                                uint8_t chip_select) except -1:
        cdef pair[int,int] key = pair[int,int](bus_id, chip_select)
        if self.dds_chn_map.count(key) != 0:
            return self.dds_chn_map[key]
        cdef DDSChannel dds_chn
        dds_chn.ftw_per_hz = ftw_per_hz
        dds_chn.bus_id = bus_id
        dds_chn.chip_select = chip_select
        cdef int dds_id = self.ddschns.size()
        self.ddschns.push_back(dds_chn)
        self.dds_chn_map[key] = dds_id
        return dds_id

    cdef int add_dds_param_channel(self, int seqchn, int dds_id,
                                   ChannelType param) except -1:
        assert self.dds_param_chn_map.count(seqchn) == 0
        self.dds_param_chn_map[seqchn] = pair[int,ChannelType](dds_id, param)
        return 0

    cdef get_device(self, str name):
        if hasattr(self.sys, 'registry'):
            # DAX support
            unique = <str?>self.sys.registry.get_unique_device_key(name)
        else:
            unique = name
        # Do not call the get_device function from DAX since
        # it assumes that the calling object will take ownership of the deivce.
        cls = artiq.language.environment.HasEnvironment
        return cls.get_device(self.sys, unique)

    cdef int add_channel_artiq(self, int idx, tuple path) except -1:
        dev = self.get_device(<str>path[1])
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
                self.add_ttl_channel(idx, <int?>dev.sw.target_o, False)
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
            dds_id = self.get_dds_channel_id(bus_id, <double?>dev.ftw_per_hz,
                                             <int?>dev.chip_select)
            self.add_dds_param_channel(idx, dds_id, dds_param_type)
        elif isinstance(dev, DevTTLOut):
            if len(path) > 2:
                raise_invalid_channel(path)
            self.add_ttl_channel(idx, <int?>dev.target_o, False)
        elif isinstance(dev, DevEdgeCounter):
            if len(path) > 2:
                raise_invalid_channel(path)
            self.add_ttl_channel(idx, (<int?>dev.channel) << 8, True)
        else:
            devstr = str(dev)
            PyErr_Format(ValueError, 'Unsupported device: %U', <PyObject*>devstr)
        return 0

    cdef int collect_channels(self, Seq seq) except -1:
        cdef int idx = -1
        for _path in seq.seqinfo.channel_paths:
            idx += 1
            path = <tuple>_path
            if <str>path[0] != 'artiq':
                continue
            if len(path) < 2:
                raise_invalid_channel(path)
            self.add_channel_artiq(idx, path)
        self.dds_chn_map.clear() # Not needed after channel collection
        return 0

cdef ChannelsInfo new_channels_info(sys):
    self = <ChannelsInfo>ChannelsInfo.__new__(ChannelsInfo)
    self.sys = sys
    return self

@cython.final
cdef class ArtiqBackend:
    def __init__(self):
        PyErr_Format(TypeError, "ArtiqBackend cannot be created directly")

    cdef int process_seq(self) except -1:
        self.channels.collect_channels(self.seq)
        # TODO

cpdef ArtiqBackend new_artiq_backend(sys, Seq seq):
    self = <ArtiqBackend>ArtiqBackend.__new__(ArtiqBackend)
    self.seq = seq
    self.channels = new_channels_info(sys)
    return self

#     def get_data1(self):
#         def _get_data1(amp, phase):
#             asf = np.int32(round(amp * 0x3fff))
#             pow_ = np.int32(round(phase * 0x10000) & 0xffff)
#             return (asf << 16) | pow_

#     def get_data2(self):
#         def _get_data2(freq):
#             return np.int32(round(self._ftw_per_hz * freq))

# class DDSAction:
#     __slots__ = ['bus', 'cs', 'data1', 'data2', 'exact_time', 'disabled']
#     def __init__(self, bus, cs, data1, data2, exact_time):
#         self.bus = bus
#         self.cs = cs
#         self.data1 = data1
#         self.data2 = data2
#         self.exact_time = exact_time
#         # Record at runtime if the previous run had this action disabled
#         self.disabled = False

# class TTLAction:
#     __slots__ = ['target', 'value', 'iscounter', 'exact_time', 'disabled']
#     def __init__(self, target, value, exact_time, iscounter):
#         self.target = target
#         self.value = value
#         self.iscounter = iscounter
#         self.exact_time = exact_time
#         # Record at runtime if the previous run had this action disabled
#         self.disabled = False


# class RTIOScheduler:
#         time_mu = 0
#         for timed_action in self.timed_actions:
#             actual_time_mu, target, value = timed_action
#             rtio_output_idx = self.add_rtio_wait(rtio_output_idx,
#                                                  actual_time_mu - time_mu)
#             time_mu = actual_time_mu
#             if rtio_output_idx + 1 >= len(self.rtio_output):
#                 self.rtio_output.resize(rtio_output_idx * 2 + 2, refcheck=False)
#             self.rtio_output[rtio_output_idx] = target
#             self.rtio_output[rtio_output_idx + 1] = value
#             rtio_output_idx += 2

#         rtio_output_idx = self.add_rtio_wait(rtio_output_idx, total_time_mu - time_mu)
#         self.rtio_output.resize(rtio_output_idx, refcheck=False)

#     def add_rtio_wait(self, rtio_output_idx, t_mu):
#         if t_mu > 0x8000_000:
#             nwait_eles = t_mu // 0x8000_000
#             t_mu = t_mu % 0x8000_000
#             if len(self.rtio_output) < (rtio_output_idx + nwait_eles):
#                 self.rtio_output.resize((rtio_output_idx + nwait_eles) * 2 + 2,
#                                         refcheck=False)
#             self.rtio_output[rtio_output_idx:rtio_output_idx + nwait_eles] = 0x8000_000
#             rtio_output_idx += nwait_eles
#         if t_mu > 0:
#             if rtio_output_idx >= len(self.rtio_output):
#                 self.rtio_output.resize(rtio_output_idx * 2 + 2, refcheck=False)
#             self.rtio_output[rtio_output_idx] = -t_mu
#             rtio_output_idx += 1
#         return rtio_output_idx


# class ArtiqBackend:
#     def add_start_trigger_ttl(self, tgt, time, min_time, raising_edge):
#         time_mu = 3000 + int(time / 1000)
#         min_time_mu = max(int(min_time / 1000), 8)
#         start_time_mu = time_mu if raising_edge else time_mu - min_time_mu
#         if start_time_mu < 0:
#             raise ValueError("Start trigger time too early")
#         self.start_trigger_ttls.append((tgt, time_mu, min_time_mu, raising_edge))

#     def process_seq(self):
#         self.collect_actions()
#         self.populate_scheduler()

#     def collect_actions(self):
#         # Assume the time tuple all share the same prefix (there's a single time line)
#         prev_time_idx = 0
#         prev_time_seq = 3e6 # 3 us start offset
#         prev_time_mu = 3000
#         def update_time(time):
#             nonlocal prev_time_idx
#             nonlocal prev_time_seq
#             nonlocal prev_time_mu
#             timelen = len(time)
#             if timelen == prev_time_idx:
#                 return
#             assert timelen > prev_time_idx
#             if timelen == prev_time_idx + 1:
#                 new_time = time[prev_time_idx]
#             else:
#                 new_time = sum(time[prev_time_idx:])
#             prev_time_idx = timelen
#             prev_time_seq = prev_time_seq + new_time
#             # Hard code the 1000 for now...
#             prev_time_mu = round_time(prev_time_seq / 1000)
#             self.all_times_mu[prev_time_idx] = prev_time_mu

#         pending_actions_time = ()
#         pending_ttl_actions = {}
#         pending_dds_actions = {}
#         def flush_pending_actions():
#             if not pending_ttl_actions and not pending_dds_actions:
#                 return
#             update_time(pending_actions_time)
#             for actions in pending_ttl_actions.values():
#                 self.actions.append((prev_time_idx, actions))
#             pending_ttl_actions.clear()
#             for actions in pending_dds_actions.values():
#                 self.actions.append((prev_time_idx, actions))
#             pending_dds_actions.clear()
#         def queue_ttl_action(chn, action):
#             old_action = pending_ttl_actions.get(chn)
#             if old_action is not None and old_action.exact_time:
#                 action.exact_time = True
#             pending_ttl_actions[chn] = action
#         def queue_dds_action(chn, action):
#             old_action = pending_dds_actions.get(chn)
#             if old_action is not None and old_action.exact_time:
#                 action.exact_time = True
#             pending_dds_actions[chn] = action

#         dds_status = {}
#         for (chn_addr, cs, ftw_per_hz, arg_type) in self.dds_chns.values():
#             if (chn_addr, cs) not in dds_status:
#                 dds_status[(chn_addr, cs)] = AD9910Status(ftw_per_hz)

#         def get_dds_chn_info(chn):
#             chn_addr, cs, ftw_per_hz, arg_type = self.dds_chns[chn]
#             status = dds_status[(chn_addr, cs)]
#             return status, arg_type

#         pulse_actions = []
#         for step in self.seq.subseqs:
#             if len(step.toffset) > len(pending_actions_time):
#                 flush_pending_actions()
#                 pending_actions_time = step.toffset
#             for (chn, action) in step.actions.items():
#                 if chn in self.ttls:
#                     tgt, iscounter = self.ttls[chn]
#                     value = action.value
#                     if iscounter:
#                         if isinstance(value, RuntimeValue):
#                             raise ValueError("Edge counter value cannot be dynamic")
#                         value = COUNTER_ENABLE if value else COUNTER_DISABLE
#                     queue_ttl_action(chn, TTLAction(tgt, value, action.exact_time,
#                                                     iscounter))
#                 elif chn in self.dds_chns:
#                     status, arg_type = get_dds_chn_info(chn)
#                     if arg_type == 'amp':
#                         status.set_amp(action.value)
#                     elif arg_type == 'freq':
#                         status.set_freq(action.value)
#                     elif arg_type == 'phase':
#                         status.set_phase(action.value)
#                     else:
#                         assert False
#                     if status.dirty:
#                         # If the action didn't do anything,
#                         # don't set the exact time flag either.
#                         # so that it won't be applied to the next pulse.
#                         status.set_exact_time(action.exact_time)
#                 else:
#                     continue
#                 if action.kws:
#                     raise ValueError(f'Invalid output keyword argument {action.kws}')
#                 if action.is_pulse:
#                     pulse_actions.append((chn, action))

#             for ((chn_addr, cs), status) in dds_status.items():
#                 if not status.dirty:
#                     continue
#                 bus = self.urukul_buses[chn_addr]
#                 queue_dds_action((chn_addr, cs),
#                                  DDSAction(bus, cs, status.get_data1(),
#                                            status.get_data2(),
#                                            status.exact_time))
#                 status.clear_action()

#             if not pulse_actions:
#                 continue

#             if isinstance(step.tlen, RuntimeValue) or step.tlen != 0:
#                 flush_pending_actions()
#                 pending_actions_time = (*step.toffset, step.tlen)

#             for (chn, action) in pulse_actions:
#                 if chn in self.ttls:
#                     tgt, iscounter = self.ttls[chn]
#                     value = action.end_val
#                     if iscounter:
#                         if isinstance(value, RuntimeValue):
#                             raise ValueError("Edge counter value cannot be dynamic")
#                         value = COUNTER_ENABLE if value else COUNTER_DISABLE
#                     queue_ttl_action(chn, TTLAction(tgt, value,
#                                                     action.exact_time, iscounter))
#                 else: # assert chn in self.dds_chns:
#                     status, arg_type = get_dds_chn_info(chn)
#                     value = action.end_val
#                     if arg_type == 'amp':
#                         status.set_amp(value)
#                     elif arg_type == 'freq':
#                         status.set_freq(value)
#                     elif arg_type == 'phase':
#                         status.set_phase(value)
#                     else:
#                         assert False
#                     if status.dirty:
#                         # If the action didn't do anything,
#                         # don't set the exact time flag either.
#                         # so that it won't be applied to the next pulse.
#                         status.set_exact_time(action.exact_time)

#             pulse_actions.clear()

#             for ((chn_addr, cs), status) in dds_status.items():
#                 if not status.dirty:
#                     continue
#                 bus = self.urukul_buses[chn_addr]
#                 queue_dds_action((chn_addr, cs),
#                                  DDSAction(bus, cs, status.get_data1(),
#                                            status.get_data2(),
#                                            status.exact_time))
#                 status.clear_action()

#         flush_pending_actions()
#         update_time(self.seq.time_segments)
#         self.total_time_idx = prev_time_idx

#     def populate_scheduler(self):
#         ttl_queues = {}
#         ddsbus_queues = {}

#         rt_value_map = self.rt_value_map
#         rt_time_map = self.rt_time_map

#         action_map = self.action_map

#         rtio_actions = self.scheduler.rtio_actions

#         for (target, time_mu, min_time_mu, raising_edge) in self.start_trigger_ttls:
#             idx0 = len(rtio_actions)
#             rtio_actions.append(RTIOAction(target, 1, time_mu if raising_edge else -1,
#                                            -1, 0, raising_edge, False))
#             rtio_actions.append(RTIOAction(target, 0, -1 if raising_edge else time_mu,
#                                            idx0, min_time_mu, not raising_edge, False))

#         for action_idx, (time_idx, action) in enumerate(self.actions):
#             time_mu = self.all_times_mu[time_idx]
#             if isinstance(time_mu, RuntimeValue):
#                 rt_times = rt_time_map.get(time_idx)
#                 if rt_times is None:
#                     rt_times = []
#                 rt_time_map[time_idx] = rt_times
#                 time_mu = 0
#             else:
#                 rt_times = None
#             rtio_idx = len(rtio_actions)
#             if isinstance(action, TTLAction):
#                 target = action.target
#                 last_idx, last_gap = ttl_queues.get(target, (-1, 0))
#                 ttl_queues[target] = (rtio_idx, 1)
#                 value = action.value
#                 if isinstance(value, RuntimeValue):
#                     assert not action.iscounter
#                     rt_value_map[rtio_idx] = (value != 0).eval
#                     value = 0
#                 elif not action.iscounter:
#                     value = 1 if value else 0
#                 if rt_times is not None:
#                     rt_times.append(rtio_idx)
#                 action_map[action_idx] = [rtio_idx]
#                 rtio_actions.append(RTIOAction(target, value, time_mu,
#                                                last_idx, last_gap,
#                                                action.exact_time, False))
#             elif isinstance(action, DDSAction):
#                 ios = action.bus.write_set(action.cs, action.data1, action.data2)
#                 rtio_list = []
#                 action_map[action_idx] = rtio_list
#                 bus = action.bus
#                 channel = bus.channel

#                 last_idx, last_gap = ddsbus_queues.get(channel, (-1, 0))
#                 for io in ios:
#                     data = io[1]
#                     if callable(data):
#                         rt_value_map[rtio_idx] = data
#                         data = 0
#                     rtio_actions.append(RTIOAction(io[0], data, -1, last_idx,
#                                                    last_gap, False, True))
#                     last_idx, last_gap = rtio_idx, io[2]
#                     rtio_list.append(rtio_idx)
#                     rtio_idx += 1
#                 if rt_times is not None:
#                     rt_times.append(rtio_idx)
#                 rtio_list.append(rtio_idx)
#                 rtio_actions.append(RTIOAction(bus.io_update_target, 1, time_mu,
#                                                last_idx, last_gap,
#                                                action.exact_time, False))
#                 last_idx, last_gap = rtio_idx, 8
#                 rtio_idx += 1
#                 rtio_list.append(rtio_idx)
#                 rtio_actions.append(RTIOAction(bus.io_update_target, 0, -1,
#                                                last_idx, last_gap, False, False))
#                 ddsbus_queues[channel] = rtio_idx, 8

#     def get_rtios(self, age):
#         rtio_actions = self.scheduler.rtio_actions
#         for (rtio_idx, cb) in self.rt_value_map.items():
#             rtio_actions[rtio_idx].value = cb(age)
#         for (time_idx, rt_times) in self.rt_time_map.items():
#             time_mu = self.all_times_mu[time_idx].eval(age)
#             for rtio_idx in rt_times:
#                 rtio_actions[rtio_idx].request_time_mu = time_mu

#         # XXX: if we decided to merge a exact time pulse with a non-exact time pulse
#         # we currently do not propagate the exact time flag to the non-exact one.

#         all_times_mu = {idx: t_mu.eval(age) if isinstance(t_mu, RuntimeValue) else t_mu
#                             for (idx, t_mu) in self.all_times_mu.items()}

#         last_actions = {}

#         actions = self.actions
#         nactions = len(actions)
#         for action_idx in range(nactions - 1, -1, -1):
#             tidx, action = actions[action_idx]
#             t_mu = all_times_mu[tidx]
#             disable_unallowed = False
#             if isinstance(action, TTLAction):
#                 key = action.target, 0
#                 min_gap = 8 if action.iscounter else 1
#                 disable_unallowed = action.iscounter
#             elif isinstance(action, DDSAction):
#                 key = action.bus.channel, action.cs
#                 min_gap = 1300
#             last_t_mu = last_actions.get(key)
#             disabled = False
#             if last_t_mu is None or last_t_mu > t_mu + min_gap:
#                 last_actions[key] = t_mu
#             elif disable_unallowed:
#                 raise ValueError('No time for counter config')
#             else:
#                 disabled = True
#             if action.disabled == disabled:
#                 continue
#             action.disabled = disabled
#             for rtio_idx in self.action_map[action_idx]:
#                 rtio_actions[rtio_idx].disabled = disabled

#         total_time_mu = all_times_mu[self.total_time_idx]
#         err, rtio_idx = self.scheduler.schedule_actions(total_time_mu)
#         if err != ScheduleErrorCode.NoError:
#             # TODO: more efficient lookup
#             for action_idx, rtio_idxs in self.action_map.items():
#                 if rtio_idx in rtio_idxs:
#                     raise ArtiqSchedulingError(err, rtio_idx, action_idx)
#         return self.scheduler.rtio_output
