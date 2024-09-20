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
  PyExc_TypeError, PyExc_ValueError, _PyObject_Vectorcall, pyfloat_from_double

cimport cython
from cython.operator cimport dereference as deref
from cpython cimport PyObject, Py_LT, Py_GT, PyFloat_AS_DOUBLE

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
cdef dummy_spline_segments = []

cdef class RampFunction:
    def __init__(self, *, **params):
        for (name, value) in params.items():
            setattr(self, name, value)

    cdef eval(self, t, length, oldval):
        cdef PyObject *_eval = <PyObject*>self._eval
        if <object>_eval is None:
            self._eval = type(self).eval
            _eval = <PyObject*>self._eval
        cdef PyObject *args[4]
        args[0] = <PyObject*>self
        args[1] = <PyObject*>t
        args[2] = <PyObject*>length
        args[3] = <PyObject*>oldval
        return _PyObject_Vectorcall(_eval, args, 4, NULL)

    cdef spline_segments(self, double length, double oldval):
        cdef PyObject *_spline_segments = <PyObject*>self._spline_segments
        if <object>_spline_segments is None:
            try:
                self._spline_segments = type(self).spline_segments
            except AttributeError:
                self._spline_segments = dummy_spline_segments
            _spline_segments = <PyObject*>self._spline_segments
        if <object>_spline_segments is dummy_spline_segments:
            return
        cdef PyObject *args[3]
        cdef object pylen = pyfloat_from_double(length)
        cdef object pyoldval = pyfloat_from_double(oldval)
        args[0] = <PyObject*>self
        args[1] = <PyObject*>pylen
        args[2] = <PyObject*>pyoldval
        return _PyObject_Vectorcall(_spline_segments, args, 3, NULL)

    cdef int set_compile_params(self, length, oldval) except -1:
        fvalue = self.eval(arg0, length, oldval)
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

    cdef int set_runtime_params(self, unsigned age, py_object &pyage) except -1:
        if self.interp_func:
            interp_function_eval_all(deref(self.interp_func), age, pyage)

    cdef TagVal runtime_eval(self, double t) noexcept:
        cdef void *fvalue
        if not self.interp_func:
            # Avoid reference counting
            fvalue = <void*>self._fvalue
            return TagVal(PyFloat_AS_DOUBLE(<object>fvalue))
        rampfunc_set_time(self, t)
        return interp_function_call(deref(self.interp_func))

@cython.final
cdef class SeqCubicSpline:
    @property
    def order0(self):
        return self._spline_segments

    @property
    def order1(self):
        return self._fvalue

    def __init__(self, order0, order1=0.0, order2=0.0, order3=0.0):
        self._spline_segments = order0
        self._fvalue = order1
        self.order2 = order2
        self.order3 = order3

    cdef eval(self, t, length, oldval):
        t = t / length
        return self._spline_segments + (self._fvalue + (self.order2 + self.order3 * t) * t) * t

    cdef spline_segments(self, double length, double oldval):
        return ()

    cdef int set_compile_params(self, length, oldval) except -1:
        self._eval = length

    cdef int set_runtime_params(self, unsigned age, py_object &pyage) except -1:
        self.f_order0 = get_value_f64(self._spline_segments, age, pyage)
        self.f_order1 = get_value_f64(self._fvalue, age, pyage)
        self.f_order2 = get_value_f64(self.order2, age, pyage)
        self.f_order3 = get_value_f64(self.order3, age, pyage)
        self.f_length = get_value_f64(self._eval, age, pyage)

    @cython.cdivision(True)
    cdef TagVal runtime_eval(self, double t) noexcept:
        t = t / self.f_length
        return TagVal(self.f_order0 + (self.f_order1 +
                                       (self.f_order2 + self.f_order3 * t) * t) * t)

# These can be implemented in python code but are provided here
# to be slightly more efficient.
cdef np_cos = np.cos
cdef m_pi = cmath.pi
cdef m_2pi = cmath.pi * 2

@cython.final
cdef class Blackman(RampFunction):
    # _eval -> length
    # _spline_segments -> amp
    # _fvalue -> offset
    @property
    def amp(self):
        return self._spline_segments

    @property
    def offset(self):
        return self._fvalue

    cdef double f_amp
    cdef double f_offset
    cdef double f_length
    def __init__(self, amp, offset=0):
        self._spline_segments = amp
        self._fvalue = offset

    cdef eval(self, t, length, oldval):
        if not is_rtval(length) and length == 0:
            val = 0.0
        else:
            theta = t * (m_2pi / length) - m_pi
            cost = np_cos(theta)
            val = self._spline_segments * (0.34 + cost * (0.5 + 0.16 * cost))
            val = ifelse(length == 0, 0.0, val)
        return val + self._fvalue

    cdef spline_segments(self, double length, double oldval):
        pass

    cdef int set_compile_params(self, length, oldval) except -1:
        self._eval = length

    cdef int set_runtime_params(self, unsigned age, py_object &pyage) except -1:
        self.f_length = get_value_f64(self._eval, age, pyage)
        self.f_amp = get_value_f64(self._spline_segments, age, pyage)
        self.f_offset = get_value_f64(self._fvalue, age, pyage)

    @cython.cdivision(True)
    cdef TagVal runtime_eval(self, double t) noexcept:
        if self.f_length == 0:
            return TagVal(self.f_offset)
        theta = t * (cmath.pi * 2 / self.f_length) - cmath.pi
        cost = cmath.cos(theta)
        val = self.f_amp * (0.34 + cost * (0.5 + 0.16 * cost))
        return TagVal(val + self.f_offset)

@cython.final
cdef class BlackmanSquare(RampFunction):
    # _eval -> length
    # _spline_segments -> amp
    # _fvalue -> offset
    @property
    def amp(self):
        return self._spline_segments

    @property
    def offset(self):
        return self._fvalue

    cdef double f_amp
    cdef double f_offset
    cdef double f_length
    def __init__(self, amp, offset=0):
        self._spline_segments = amp
        self._fvalue = offset

    cdef eval(self, t, length, oldval):
        if not is_rtval(length) and length == 0:
            val = 0.0
        else:
            theta = t * (m_2pi / length) - m_pi
            cost = np_cos(theta)
            val = 0.34 + cost * (0.5 + 0.16 * cost)
            val = self._spline_segments * val * val
            val = ifelse(length == 0, 0.0, val)
        return val + self._fvalue

    cdef spline_segments(self, double length, double oldval):
        pass

    cdef int set_compile_params(self, length, oldval) except -1:
        self._eval = length

    cdef int set_runtime_params(self, unsigned age, py_object &pyage) except -1:
        self.f_length = get_value_f64(self._eval, age, pyage)
        self.f_amp = get_value_f64(self._spline_segments, age, pyage)
        self.f_offset = get_value_f64(self._fvalue, age, pyage)

    @cython.cdivision(True)
    cdef TagVal runtime_eval(self, double t) noexcept:
        if self.f_length == 0:
            return TagVal(self.f_offset)
        theta = t * (cmath.pi * 2 / self.f_length) - cmath.pi
        cost = cmath.cos(theta)
        cdef double val = 0.34 + cost * (0.5 + 0.16 * cost)
        val = self.f_amp * val * val
        return TagVal(val + self.f_offset)

@cython.final
cdef class LinearRamp(RampFunction):
    # _eval -> length
    # _spline_segments -> start
    # _fvalue -> end
    @property
    def start(self):
        return self._spline_segments

    @property
    def end(self):
        return self._fvalue

    cdef double f_start
    cdef double f_end
    cdef double f_length
    def __init__(self, start, end):
        self._spline_segments = start
        self._fvalue = end

    cdef eval(self, t, length, oldval):
        t = t / length
        return self._spline_segments * (1 - t) + self._fvalue * t

    cdef spline_segments(self, double length, double oldval):
        return ()

    cdef int set_compile_params(self, length, oldval) except -1:
        self._eval = length

    cdef int set_runtime_params(self, unsigned age, py_object &pyage) except -1:
        self.f_length = get_value_f64(self._eval, age, pyage)
        self.f_start = get_value_f64(self._spline_segments, age, pyage)
        self.f_end = get_value_f64(self._fvalue, age, pyage)

    @cython.cdivision(True)
    cdef TagVal runtime_eval(self, double t) noexcept:
        t = t / self.f_length
        return TagVal(self.f_start * (1 - t) + self.f_end * t)
