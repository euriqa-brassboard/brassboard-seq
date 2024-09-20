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
from brassboard_seq.rtval cimport _get_value, get_value_f64, ifelse, is_rtval, \
  new_arg, new_const, new_expr2, ValueType, DataType, rt_eval_tagval, RuntimeValue, \
  interp_function_set_value, interp_function_eval_all, interp_function_call
from brassboard_seq.utils cimport PyErr_Format, Py_NotImplemented, \
  PyExc_TypeError, PyExc_ValueError

cimport cython
from cython.operator cimport dereference as deref
from cpython cimport Py_LT, Py_GT, PyTuple_GET_ITEM, PyFloat_AS_DOUBLE

cdef np # hide import
import numpy as np

from libc cimport math as cmath
from libcpp.vector cimport vector
from libcpp.utility cimport move

cdef extern from "src/action.cpp" namespace "brassboard_seq::action":
    void rampfunc_set_time(RampFunction self, double t)

@cython.auto_pickle(False)
@cython.no_gc
@cython.final
cdef class Action:
    def __init__(self):
        PyErr_Format(PyExc_TypeError, "Action cannot be created directly")

    def __richcmp__(self, other, int op):
        # For sorting actions according to their times.
        tid1 = self.tid
        if type(other) is not Action:
            # Action is a final type so we can use direct type comparison
            return <object>Py_NotImplemented
        tid2 = (<Action>other).tid
        if op == Py_LT:
            return tid1 < tid2
        elif op == Py_GT:
            return tid1 > tid2
        return <object>Py_NotImplemented

    def __str__(self):
        name = 'Pulse' if self.data.is_pulse else 'Set'
        if self.kws is None:
            kws = ''
        else:
            kws = ''.join(f', {name}={val}' for (name, val) in self.kws.items())
        cond = self.cond
        if cond is not True:
            cond_str = f', cond={cond}'
        else:
            cond_str = ''
        if self.data.exact_time:
            return f'{name}({self.value}{cond_str}, exact_time=True{kws})'
        return f'{name}({self.value}{cond_str}{kws})'

    def __repr__(self):
        return str(self)

cdef RuntimeValue arg0 = new_arg(0)
cdef RuntimeValue const0 = new_const(0.0)

cdef class RampFunction:
    def __init__(self, *, **params):
        self.params = params
        self._eval = getattr(type(self), 'eval')
        try:
            self._spline_segments = getattr(type(self), 'spline_segments')
        except AttributeError:
            pass

cdef cubic_spline_eval
def cubic_spline_eval(SeqCubicSpline self, t, length, oldval, /):
    t = t / length
    if self.compile_mode:
        orders = <tuple>self._spline_segments
        return (<object>PyTuple_GET_ITEM(orders, 0)) + ((<object>PyTuple_GET_ITEM(orders, 1)) + ((<object>PyTuple_GET_ITEM(orders, 2)) + (<object>PyTuple_GET_ITEM(orders, 3)) * t) * t) * t
    return self.order0 + (self.order1 + (self.order2 + self.order3 * t) * t) * t

@cython.final
cdef class SeqCubicSpline:
    def __init__(self, order0, order1=0.0, order2=0.0, order3=0.0):
        self._spline_segments = (order0, order1, order2, order3)
        self._eval = cubic_spline_eval

    @property
    def order0(self):
        return <object>PyTuple_GET_ITEM(self._spline_segments, 0)

    @property
    def order1(self):
        return <object>PyTuple_GET_ITEM(self._spline_segments, 1)

    @property
    def order2(self):
        return <object>PyTuple_GET_ITEM(self._spline_segments, 2)

    @property
    def order3(self):
        return <object>PyTuple_GET_ITEM(self._spline_segments, 3)

cdef ramp_eval(RampFunction self, t, length, oldval):
    return self._eval(self, t, length, oldval)

cdef int ramp_set_compile_params(RampFunction self, length, oldval) except -1:
    if type(self) is SeqCubicSpline:
        (<SeqCubicSpline>self).compile_mode = True
        return 0
    for (name, value) in self.params.items():
        setattr(self, name, value)
    fvalue = ramp_eval(self, arg0, length, oldval)
    cdef vector[DataType] args
    cdef unique_ptr[InterpFunction] interp_func
    if is_rtval(fvalue):
        interp_func.reset(new InterpFunction())
        args.push_back(DataType.Float64)
        if (<RuntimeValue>fvalue).cache.type != DataType.Float64:
            fvalue = new_expr2(ValueType.Add, fvalue, const0)
        interp_function_set_value(deref(interp_func), <RuntimeValue>fvalue, args)
        self.interp_func = move(interp_func)
    else:
        fvalue = <double>fvalue
    self._fvalue = fvalue

cdef int ramp_set_runtime_params(RampFunction self, unsigned age,
                                 py_object &pyage) except -1:
    if type(self) is SeqCubicSpline:
        sp = <SeqCubicSpline>self
        orders = <tuple>sp._spline_segments
        sp.order0 = get_value_f64(<object>PyTuple_GET_ITEM(orders, 0), age, pyage)
        sp.order1 = get_value_f64(<object>PyTuple_GET_ITEM(orders, 1), age, pyage)
        sp.order2 = get_value_f64(<object>PyTuple_GET_ITEM(orders, 2), age, pyage)
        sp.order3 = get_value_f64(<object>PyTuple_GET_ITEM(orders, 3), age, pyage)
        sp.compile_mode = False
        return 0
    for (name, value) in self.params.items():
        setattr(self, name, _get_value(value, age, pyage))
    if self.interp_func:
        interp_function_eval_all(deref(self.interp_func), age, pyage)

cdef TagVal ramp_interp_eval(RampFunction self, double t) noexcept:
    if type(self) is SeqCubicSpline:
        sp = <SeqCubicSpline>self
        return TagVal(sp.order0 + (sp.order1 + (sp.order2 + sp.order3 * t) * t) * t)
    cdef void *fvalue
    if not self.interp_func:
        # Avoid reference counting
        fvalue = <void*>self._fvalue
        return TagVal(PyFloat_AS_DOUBLE(<object>fvalue))
    rampfunc_set_time(self, t)
    return interp_function_call(deref(self.interp_func))

# These can be implemented in python code but are provided here
# to be slightly more efficient.
cdef np_cos = np.cos
cdef m_pi = cmath.pi
cdef m_2pi = cmath.pi * 2
cdef blackman_eval
def blackman_eval(Blackman self, t, length, oldval, /):
    if not is_rtval(length) and length == 0:
        val = t * 0
    else:
        theta = t * (m_2pi / length) - m_pi
        cost = np_cos(theta)
        val = self.amp * (0.34 + cost * (0.5 + 0.16 * cost))
        val = ifelse(length == 0, t * 0, val)
    return val + self.offset

@cython.final
cdef class Blackman(RampFunction):
    cdef public object amp
    cdef public object offset
    def __init__(self, amp, offset=0):
        self.params = {'amp': amp, 'offset': offset}
        self._eval = blackman_eval

cdef blackman_square_eval
def blackman_square_eval(BlackmanSquare self, t, length, oldval, /):
    if not is_rtval(length) and length == 0:
        val = t * 0
    else:
        theta = t * (m_2pi / length) - m_pi
        cost = np_cos(theta)
        val = 0.34 + cost * (0.5 + 0.16 * cost)
        val = self.amp * val * val
        val = ifelse(length == 0, t * 0, val)
    return val + self.offset

@cython.final
cdef class BlackmanSquare(RampFunction):
    cdef public object amp
    cdef public object offset
    def __init__(self, amp, offset=0):
        self.params = {'amp': amp, 'offset': offset}
        self._eval = blackman_square_eval

cdef linear_eval
def linear_eval(LinearRamp self, t, length, oldval, /):
    t = t / length
    return self.start * (1 - t) + self.end * t

cdef linear_spline_segments
def linear_spline_segments(self, length, oldval, /):
    return ()

@cython.final
cdef class LinearRamp(RampFunction):
    cdef public object start
    cdef public object end
    def __init__(self, start, end):
        self.params = {'start': start, 'end': end}
        self._eval = linear_eval
        self._spline_segments = linear_spline_segments
