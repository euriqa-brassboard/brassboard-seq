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
from brassboard_seq.action cimport new_action, Action, RampFunction
from brassboard_seq.config cimport translate_channel
from brassboard_seq cimport event_time
from brassboard_seq.event_time cimport is_ordered, round_time_int, round_time_rt, \
  set_base_int, set_base_rt, new_time_manager
from brassboard_seq.rtval cimport get_value_bool, ifelse, is_rtval, \
  RuntimeValue, rt_eval_tagval
from brassboard_seq.scan cimport new_param_pack
from brassboard_seq.utils cimport assume_not_none, _assume_not_none, \
  action_key, assert_key, bb_err_format, bb_raise, event_time_key, \
  new_list_of_list, set_global_tracker, pyobject_call, pytuple_prepend1, \
  PyErr_Format, PyExc_TypeError, PyExc_ValueError

cdef StringIO # hide import
from io import StringIO

cimport cython
from cpython cimport PyObject, PyDict_GetItemWithError, \
  PyList_GET_SIZE, PyList_GET_ITEM, \
  Py_INCREF, PyLong_AsLong, PyTypeObject, PyTuple_GET_ITEM

cdef extern from "src/seq.cpp" namespace "brassboard_seq::seq":
    PyTypeObject *event_time_type
    PyTypeObject *runtime_value_type
    void update_timestep(PyTypeObject*, TimeStep, RuntimeValue) except +
    void update_subseq(PyTypeObject*, SubSeq, TimeSeq, RuntimeValue) except +
    void update_conditional(PyTypeObject*, ConditionalWrapper, TimeSeq,
                            RuntimeValue) except +
    object combine_cond(object cond1, object new_cond, RuntimeValue) except +

event_time_type = <PyTypeObject*>EventTime
runtime_value_type = <PyTypeObject*>RuntimeValue

update_timestep(<PyTypeObject*>TimeStep, None, None)
update_subseq(<PyTypeObject*>SubSeq, None, None, None)
update_conditional(<PyTypeObject*>ConditionalWrapper, None, None, None)

@cython.auto_pickle(False)
cdef class TimeSeq:
    def __init__(self):
        PyErr_Format(PyExc_TypeError, "TimeSeq cannot be created directly")

    def get_channel_id(self, str name, /):
        return _get_channel_id(self.seqinfo, name)

    def set_time(self, EventTime time, /, offset=0): # offset in seconds
        if is_rtval(offset):
            set_base_rt(self.start_time, time, round_time_rt(<RuntimeValue>offset))
        else:
            set_base_int(self.start_time, time, round_time_int(offset))
        self.seqinfo.bt_tracker.record(event_time_key(<void*>self.start_time))

    def rt_assert(self, c, /, str msg="Assertion failed"):
        if is_rtval(c):
            self.seqinfo.bt_tracker.record(assert_key(len(self.seqinfo.assertions)))
            self.seqinfo.assertions.append((c, msg))
            return
        if not c:
            raise AssertionError(msg)

@cython.auto_pickle(False)
@cython.no_gc
@cython.final
cdef class TimeStep(TimeSeq):
    def __init__(self):
        PyErr_Format(PyExc_TypeError, "TimeStep cannot be created directly")

    # Methods defined in c++
    # def set(self, chn, value, /, *, cond=True, bint exact_time=False, **kws)
    # def pulse(self, chn, value, /, *, cond=True, bint exact_time=False, **kws)

    def __str__(self):
        io = StringIO()
        timestep_show(self, io.write, 0)
        return io.getvalue()

    def __repr__(self):
        return str(self)

cdef inline int timestep_set(TimeStep self, chn, value, cond, bint is_pulse,
                             bint exact_time, dict kws) except -1:
    cdef int cid
    seqinfo = self.seqinfo
    if type(chn) is int:
        lcid = PyLong_AsLong(chn)
        _assume_not_none(<void*>seqinfo.channel_paths)
        if lcid < 0 or lcid > len(seqinfo.channel_paths):
            PyErr_Format(PyExc_ValueError, "Channel id %ld out of bound", lcid)
        cid = lcid
    else:
        cid = _get_channel_id(seqinfo, chn)
    if cid >= self.actions.size():
        self.actions.resize(cid + 1)
    elif self.actions[cid].get() != NULL:
        name = '/'.join(seqinfo.channel_paths[cid])
        PyErr_Format(PyExc_ValueError,
                     "Multiple actions added for the same channel "
                     "at the same time on %U.", <PyObject*>name)
    self.seqinfo.bt_tracker.record(action_key(seqinfo.action_counter))
    action = new_action(Action, value, cond, is_pulse, exact_time, kws,
                        seqinfo.action_counter, None)
    seqinfo.action_counter += 1
    Py_INCREF(action)
    self.actions[cid].reset(<PyObject*>action)
    return 0

cdef int timestep_show(TimeStep self, write, int indent) except -1:
    write(' ' * indent)
    write(f'TimeStep({self.length})@T[{self.start_time.data.id}]')
    cond = self.cond
    if cond is not True:
        write(f' if {cond}\n')
    else:
        write('\n')
    nactions = self.actions.size()
    for chn_idx in range(nactions):
        paction = self.actions[chn_idx].get()
        if paction == NULL:
            continue
        action = <Action>paction
        chn = '/'.join(self.seqinfo.channel_paths[chn_idx])
        write(' ' * (indent + 2))
        write(f'{chn}: {str(action)}\n')
    return 0

@cython.auto_pickle(False)
@cython.final
cdef class ConditionalWrapper:
    def __init__(self):
        PyErr_Format(PyExc_TypeError, "ConditionalWrapper cannot be created directly")

    def conditional(self, cond, /):
        wrapper = <ConditionalWrapper>ConditionalWrapper.__new__(ConditionalWrapper)
        wrapper.seq = self.seq
        wrapper.cond = combine_cond(self.cond, cond, None)
        wrapper.C = self.C
        return wrapper

    def wait(self, length, /, *, cond=True):
        wait_cond(self.seq, length, combine_cond(self.cond, cond, None))

    def wait_for(self, tp, /, offset=0):
        wait_for_cond(self.seq, tp, offset, self.cond)

    # Methods defined in c++
    # def add_step(self, first_arg, /, *args, **kwargs)
    # def add_background(self, first_arg, /, *args, **kwargs)
    # def add_floating(self, first_arg, /, *args, **kwargs)
    # def add_at(self, EventTime tp, first_arg, /, *args, **kwargs)
    # def set(self, chn, value, /, *, cond=True, bint exact_time=False, **kws)

    # Shorthand for add_step of custom step. Meant to be used as decorator
    def __call__(self, cb, /):
        step = add_custom_step(self.seq, self.cond, self.seq.end_time, cb, None, None)
        self.seq.end_time = step.end_time
        return step

    def __str__(self):
        io = StringIO()
        conditionalwrapper_show(self, io.write, 0)
        return io.getvalue()

    def __repr__(self):
        return str(self)

cdef int conditionalwrapper_show(ConditionalWrapper self, write, int indent) except -1:
    write(' ' * indent)
    write(f'ConditionalWrapper({self.cond}) for\n')
    if type(self.seq) == Seq:
        return seq_show(self.seq, write, indent + 2)
    return subseq_show(self.seq, write, indent + 2)

@cython.auto_pickle(False)
cdef class SubSeq(TimeSeq):
    def __init__(self):
        PyErr_Format(PyExc_TypeError, "SubSeq cannot be created directly")

    @property
    def current_time(self):
        return self.end_time

    def conditional(self, cond, /):
        wrapper = <ConditionalWrapper>ConditionalWrapper.__new__(ConditionalWrapper)
        wrapper.seq = self
        wrapper.cond = combine_cond(self.cond, cond, None)
        wrapper.C = self.C
        return wrapper

    def wait(self, length, /, *, cond=True):
        wait_cond(self, length, combine_cond(self.cond, cond, None))

    def wait_for(self, tp, /, offset=0):
        wait_for_cond(self, tp, offset, self.cond)

    # Methods defined in c++
    # def add_step(self, first_arg, /, *args, **kwargs)
    # def add_background(self, first_arg, /, *args, **kwargs)
    # def add_floating(self, first_arg, /, *args, **kwargs)
    # def add_at(self, EventTime tp, first_arg, /, *args, **kwargs)
    # def set(self, chn, value, /, *, cond=True, bint exact_time=False, **kws)

    def __str__(self):
        io = StringIO()
        subseq_show(self, io.write, 0)
        return io.getvalue()

    def __repr__(self):
        return str(self)

cdef int wait_cond(SubSeq self, length, cond) except -1:
    self.end_time = self.seqinfo.time_mgr.new_round_time(self.end_time, length,
                                                         cond, None)
    self.seqinfo.bt_tracker.record(event_time_key(<void*>self.end_time))
    return 0

cdef inline SubSeq add_custom_step(SubSeq self, cond, EventTime start_time, cb,
                                   tuple args, dict kwargs):
    subseq = <SubSeq>SubSeq.__new__(SubSeq)
    init_subseq(subseq, self, start_time, cond)
    subseq.C = self.C
    if kwargs is not None:
        pyobject_call(cb, pytuple_prepend1(args, subseq), kwargs)
    elif args is not None:
        pyobject_call(cb, pytuple_prepend1(args, subseq))
    else:
        cb(subseq)
    _assume_not_none(<void*>self.sub_seqs)
    self.sub_seqs.append(subseq)
    return subseq

cdef inline TimeStep add_time_step(SubSeq self, cond, EventTime start_time, length):
    step = <TimeStep>TimeStep.__new__(TimeStep)
    init_timeseq(step, self, start_time, cond)
    step.C = self.C
    step.length = length
    step.end_time = self.seqinfo.time_mgr.new_round_time(start_time, length,
                                                         cond, None)
    self.seqinfo.bt_tracker.record(event_time_key(<void*>step.end_time))
    _assume_not_none(<void*>self.sub_seqs)
    self.sub_seqs.append(step)
    return step

cdef int wait_for_cond(SubSeq self, _tp0, offset, cond) except -1:
    cdef EventTime tp0
    if type(_tp0) is EventTime:
        tp0 = <EventTime>_tp0
    else:
        tp0 = (<TimeSeq?>_tp0).end_time
    self.end_time = self.seqinfo.time_mgr.new_round_time(self.end_time, offset,
                                                         cond, tp0)
    self.seqinfo.bt_tracker.record(event_time_key(<void*>self.end_time))
    return 0

cdef inline int subseq_set(SubSeq self, chn, value, cond,
                           bint exact_time, dict kws) except -1:
    step = self.dummy_step
    start_time = self.end_time
    if step is None or step.end_time is not start_time:
        step = add_time_step(self, self.cond, start_time, 0)
        self.dummy_step = step
        # Update the current time so that a normal step added later
        # this is treated as ordered after this set event
        # rather than at the same time.
        self.end_time = step.end_time
    return timestep_set(step, chn, value, cond, False, exact_time, kws)

cdef int subseq_show_subseqs(SubSeq self, write, int indent) except -1:
    for _subseq in self.sub_seqs:
        subseq = <TimeSeq>_subseq
        if type(subseq) is TimeStep:
            timestep_show(<TimeStep>subseq, write, indent)
        else:
            subseq_show(<SubSeq>subseq, write, indent)
    return 0

cdef int subseq_show(SubSeq self, write, int indent) except -1:
    write(' ' * indent)
    write(f'SubSeq@T[{self.start_time.data.id}] - T[{self.end_time.data.id}]')
    cond = self.cond
    if cond is not True:
        write(f' if {cond}\n')
    else:
        write('\n')
    subseq_show_subseqs(self, write, indent + 2)
    return 0

cdef int collect_actions(SubSeq self, list actions) except -1:
    _assume_not_none(<void*>self.sub_seqs)
    for _subseq in self.sub_seqs:
        subseq = <TimeSeq>_subseq
        if type(subseq) is not TimeStep:
            collect_actions(<SubSeq>subseq, actions)
            continue
        step = <TimeStep>subseq
        tid = step.start_time.data.id
        end_tid = step.end_time.data.id
        length = step.length
        nactions = step.actions.size()
        for chn in range(nactions):
            paction = step.actions[chn].get()
            if paction == NULL:
                continue
            action = <Action>paction
            action.tid = tid
            action.end_tid = end_tid
            action.length = length
            _actions = <list>PyList_GET_ITEM(actions, chn)
            assume_not_none(_actions)
            _actions.append(action)
    return 0

@cython.auto_pickle(False)
@cython.no_gc
@cython.final
cdef class SeqInfo:
    def __init__(self):
        PyErr_Format(PyExc_TypeError, "SeqInfo cannot be created directly")

cdef int _get_channel_id(SeqInfo self, str name) except -1:
    channel_name_map = self.channel_name_map
    cdef PyObject *chnp = PyDict_GetItemWithError(channel_name_map, name)
    if chnp != NULL:
        return <int><object>chnp
    path = translate_channel(self.config, name)
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

@cython.auto_pickle(False)
@cython.final
cdef class Seq(SubSeq):
    def __init__(self, Config config, /, int max_frame=0):
        init_subseq(self, None, None, True)
        self.C = new_param_pack({}, {}, 'root')
        seqinfo = <SeqInfo>SeqInfo.__new__(SeqInfo)
        seqinfo.config = config
        seqinfo.time_mgr = new_time_manager()
        seqinfo.assertions = []
        seqinfo.bt_tracker.max_frame = max_frame
        seqinfo.channel_name_map = {}
        seqinfo.channel_path_map = {}
        seqinfo.channel_paths = []
        seqinfo.action_counter = 0
        self.seqinfo = seqinfo
        self.end_time = seqinfo.time_mgr.new_time_int(None, 0, False, True, None)
        self.seqinfo.bt_tracker.record(event_time_key(<void*>self.end_time))

    def __str__(self):
        io = StringIO()
        seq_show(self, io.write, 0)
        return io.getvalue()

    def __repr__(self):
        return str(self)

    cdef int finalize(self) except -1:
        bt_guard = set_global_tracker(&self.seqinfo.bt_tracker)
        seqinfo = self.seqinfo
        time_mgr = seqinfo.time_mgr
        time_mgr.finalize()
        seqinfo.channel_name_map = None # Free up memory
        cdef int nchn = PyList_GET_SIZE(seqinfo.channel_paths)
        all_actions = new_list_of_list(nchn)
        collect_actions(self, all_actions)
        event_times = time_mgr.event_times
        cdef EventTime last_time
        cdef bint last_is_start
        cdef int cid = -1
        cdef int tid
        assume_not_none(all_actions)
        for _actions in all_actions:
            cid += 1
            actions = <list>_actions
            assume_not_none(actions)
            actions.sort()
            value = 0
            last_time = None
            last_is_start = False
            assume_not_none(actions)
            tid = -1
            for _action in actions:
                action = <Action>_action
                if action.tid == tid:
                    # It is difficult to decide the ordering of actions
                    # if multiple were added to exactly the same time points.
                    # We disallow this in the same timestep and we'll also disallow
                    # this here.
                    name = '/'.join(seqinfo.channel_paths[cid])
                    bb_err_format(ValueError, action_key(action.aid),
                                  "Multiple actions added for the same channel "
                                  "at the same time on %U.", <PyObject*>name)
                tid = action.tid
                start_time = <EventTime>PyList_GET_ITEM(event_times, tid)
                if last_time is not None:
                    o = is_ordered(last_time, start_time)
                    if (o != event_time.OrderBefore and
                        (o != event_time.OrderEqual or last_is_start)):
                        name = '/'.join(seqinfo.channel_paths[cid])
                        bb_err_format(ValueError, action_key(action.aid),
                                      "Actions on %U is not statically ordered",
                                      <PyObject*>name)
                action.prev_val = value
                action_value = action.value
                isramp = isinstance(action_value, RampFunction)
                if not action.data.is_pulse:
                    last_is_start = not isramp
                    if action.cond is False:
                        if isramp:
                            last_time = <EventTime>PyList_GET_ITEM(event_times,
                                                                   action.end_tid)
                        else:
                            last_time = start_time
                    elif isramp:
                        rampf = <RampFunction>action_value
                        try:
                            length = action.length
                            rampf.set_compile_params(length, value)
                            new_value = rampf.eval(length, length, value)
                        except Exception as ex:
                            bb_raise(ex, action_key(action.aid))
                        value = ifelse(action.cond, new_value, value)
                        last_time = <EventTime>PyList_GET_ITEM(event_times,
                                                               action.end_tid)
                    else:
                        value = ifelse(action.cond, action_value, value)
                        last_time = start_time
                else:
                    if action.cond is not False and isramp:
                        rampf = <RampFunction>action_value
                        try:
                            rampf.set_compile_params(action.length, value)
                        except Exception as ex:
                            bb_raise(ex, action_key(action.aid))
                    last_time = <EventTime>PyList_GET_ITEM(event_times, action.end_tid)
                    last_is_start = False
                action.end_val = value
        self.all_actions = all_actions
        return 0

    cdef int runtime_finalize(self, unsigned age, py_object &pyage) except -1:
        bt_guard = set_global_tracker(&self.seqinfo.bt_tracker)
        time_mgr = self.seqinfo.time_mgr
        self.total_time = time_mgr.compute_all_times(age, pyage)
        _assume_not_none(<void*>self.seqinfo.assertions)
        cdef int assert_id = 0
        for _a in self.seqinfo.assertions:
            a = <tuple>_a
            c = <RuntimeValue>PyTuple_GET_ITEM(a, 0)
            try:
                rt_eval_tagval(c, age, pyage)
            except Exception as ex:
                bb_raise(ex, assert_key(assert_id))
            if c.cache.is_zero():
                bb_raise(AssertionError(<object>PyTuple_GET_ITEM(a, 1)),
                         assert_key(assert_id))
            assert_id += 1
        cdef long long prev_time
        cdef bint cond_val
        cdef bint is_ramp
        _assume_not_none(<void*>self.all_actions)
        for _actions in self.all_actions:
            actions = <list>_actions
            prev_time = 0
            assume_not_none(actions)
            for _action in actions:
                action = <Action>_action
                try:
                    cond_val = get_value_bool(action.cond, age, pyage)
                except Exception as ex:
                    bb_raise(ex, action_key(action.aid))
                action.data.cond_val = cond_val
                if not cond_val:
                    continue
                action_value = action.value
                is_ramp = isinstance(action_value, RampFunction)
                if is_ramp:
                    (<RampFunction>action_value).set_runtime_params(age, pyage)
                start_time = time_mgr.time_values[action.tid]
                end_time = time_mgr.time_values[action.end_tid]
                if prev_time > start_time or start_time > end_time:
                    bb_err_format(ValueError, action_key(action.aid),
                                  "Action time order violation")
                if is_ramp or action.data.is_pulse:
                    prev_time = end_time
                else:
                    prev_time = start_time
        return 0

cdef int seq_show(Seq self, write, int indent) except -1:
    write(' ' * indent)
    write(f'Seq - T[{self.end_time.data.id}]\n')
    cdef int i = 0
    for t in self.seqinfo.time_mgr.event_times:
        write(' ' * (indent + 1))
        write(f'T[{i}]: ')
        write(str(t))
        write('\n')
        i += 1
    return subseq_show_subseqs(self, write, indent + 2)

cdef inline void init_timeseq(TimeSeq self, SubSeq parent,
                              EventTime start_time, cond) noexcept:
    if parent is not None:
        self.seqinfo = parent.seqinfo
    self.start_time = start_time
    self.cond = cond

cdef inline void init_subseq(SubSeq self, SubSeq parent, EventTime start_time, cond) noexcept:
    init_timeseq(self, parent, start_time, cond)
    self.end_time = start_time
    self.sub_seqs = []
