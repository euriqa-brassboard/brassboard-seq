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
from brassboard_seq.rtval cimport get_value_f64, is_rtval, \
  new_arg, new_const, new_expr2, ValueType, DataType, RuntimeValue
from brassboard_seq.utils cimport PyObject_Vectorcall, new_float, \
  PyErr_Format, PyExc_RuntimeError

cimport cython
from cython.operator cimport dereference as deref
from cpython cimport PyFloat_AS_DOUBLE

from libc cimport math as cmath
from libcpp.vector cimport vector
from libcpp.utility cimport move

cdef extern from "src/_action.cpp" namespace "brassboard_seq::action":
    void rampfunc_set_time(RampFunction self, double t)
    void rampfunc_set_context(RampFunction self, double length, double oldval)

cdef class _RampFunctionBase:
    cdef eval_end(self, length, oldval):
        raise NotImplementedError

    cdef spline_segments(self, double length, double oldval):
        raise NotImplementedError

    cdef int set_runtime_params(self, unsigned age) except -1:
        raise NotImplementedError

    cdef TagVal runtime_eval(self, double t) noexcept:
        return TagVal()

cdef RuntimeValue arg0 = new_arg(0, float)
cdef RuntimeValue arg1 = new_arg(1, float)
cdef RuntimeValue arg2 = new_arg(2, float)
cdef RuntimeValue const0 = new_const(0.0).rel()
cdef dummy_spline_segments = []

cdef rampfunction_eval(RampFunction self, t, length, oldval):
    cdef PyObject *_eval = <PyObject*>self._eval
    if <object>_eval is None:
        self._eval = type(self).eval
        _eval = <PyObject*>self._eval
    cdef PyObject *args[4]
    args[0] = <PyObject*>self
    args[1] = <PyObject*>t
    args[2] = <PyObject*>length
    args[3] = <PyObject*>oldval
    return PyObject_Vectorcall(_eval, args, 4, NULL)

cdef class RampFunction(_RampFunctionBase):
    def __init__(self, *, **params):
        for (name, value) in params.items():
            setattr(self, name, value)
        fvalue = rampfunction_eval(self, arg0, arg1, arg2)
        cdef vector[DataType] args
        cdef unique_ptr[InterpFunction] interp_func
        if is_rtval(fvalue):
            interp_func.reset(new InterpFunction())
            args.push_back(DataType.Float64)
            args.push_back(DataType.Float64)
            args.push_back(DataType.Float64)
            if (<RuntimeValue>fvalue).datatype != DataType.Float64:
                fvalue = new_expr2(ValueType.Add, fvalue, const0).rel()
            deref(interp_func).set_value(<RuntimeValue>fvalue, args)
            self.interp_func = move(interp_func)
        elif type(fvalue) is not float:
            fvalue = new_float(<double>fvalue).rel()
        self._fvalue = fvalue

    cdef eval_end(self, length, oldval):
        return rampfunction_eval(self, length, length, oldval)

    cdef spline_segments(self, double length, double oldval):
        if self.interp_func:
            rampfunc_set_context(self, length, oldval)
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
        cdef object pylen = new_float(length).rel()
        cdef object pyoldval = new_float(oldval).rel()
        args[0] = <PyObject*>self
        args[1] = <PyObject*>pylen
        args[2] = <PyObject*>pyoldval
        return PyObject_Vectorcall(_spline_segments, args, 3, NULL)

    cdef int set_runtime_params(self, unsigned age) except -1:
        if self._fvalue is None:
            PyErr_Format(PyExc_RuntimeError, "RampFunction.__init__ not called")
        if self.interp_func:
            deref(self.interp_func).eval_all(age)

    cdef TagVal runtime_eval(self, double t) noexcept:
        cdef void *fvalue
        if not self.interp_func:
            # Avoid reference counting
            fvalue = <void*>self._fvalue
            return TagVal(PyFloat_AS_DOUBLE(<object>fvalue))
        rampfunc_set_time(self, t)
        return self.interp_func.get().call()

@cython.final
cdef class SeqCubicSpline(_RampFunctionBase):
    def __init__(self, order0, order1=0.0, order2=0.0, order3=0.0):
        self.order0 = order0
        self.order1 = order1
        self.order2 = order2
        self.order3 = order3

    cdef eval_end(self, length, oldval):
        return self.order0 + self.order1 + self.order2 + self.order3

    cdef spline_segments(self, double length, double oldval):
        self.f_inv_length = 1 / length
        return ()

    @cython.cdivision(True)
    cdef int set_runtime_params(self, unsigned age) except -1:
        self.f_order0 = get_value_f64(self.order0, age)
        self.f_order1 = get_value_f64(self.order1, age)
        self.f_order2 = get_value_f64(self.order2, age)
        self.f_order3 = get_value_f64(self.order3, age)

    @cython.cdivision(True)
    cdef TagVal runtime_eval(self, double t) noexcept:
        t = t * self.f_inv_length
        return TagVal(self.f_order0 + (self.f_order1 +
                                       (self.f_order2 + self.f_order3 * t) * t) * t)

# These ramp functions can be implemented in python code but are provided here
# to be slightly more efficient.
@cython.final
cdef class Blackman(_RampFunctionBase):
    cdef readonly object amp
    cdef readonly object offset
    cdef double f_amp
    cdef double f_offset
    cdef double f_t_scale
    def __init__(self, amp, offset=0):
        self.amp = amp
        self.offset = offset

    cdef eval_end(self, length, oldval):
        return self.offset

    cdef spline_segments(self, double length, double oldval):
        self.f_t_scale = 0.0 if length == 0 else cmath.pi * 2 / length

    @cython.cdivision(True)
    cdef int set_runtime_params(self, unsigned age) except -1:
        self.f_amp = get_value_f64(self.amp, age)
        self.f_offset = get_value_f64(self.offset, age)

    @cython.cdivision(True)
    cdef TagVal runtime_eval(self, double t) noexcept:
        cost = cmath.cos(t * self.f_t_scale)
        val = self.f_amp * (0.34 - cost * (0.5 - 0.16 * cost))
        return TagVal(val + self.f_offset)

@cython.final
cdef class BlackmanSquare(_RampFunctionBase):
    cdef readonly object amp
    cdef readonly object offset
    cdef double f_amp
    cdef double f_offset
    cdef double f_t_scale
    def __init__(self, amp, offset=0):
        self.amp = amp
        self.offset = offset

    cdef eval_end(self, length, oldval):
        return self.offset

    cdef spline_segments(self, double length, double oldval):
        self.f_t_scale = 0.0 if length == 0 else cmath.pi * 2 / length

    @cython.cdivision(True)
    cdef int set_runtime_params(self, unsigned age) except -1:
        self.f_amp = get_value_f64(self.amp, age)
        self.f_offset = get_value_f64(self.offset, age)

    @cython.cdivision(True)
    cdef TagVal runtime_eval(self, double t) noexcept:
        cost = cmath.cos(t * self.f_t_scale)
        cdef double val = 0.34 - cost * (0.5 - 0.16 * cost)
        val = self.f_amp * val * val
        return TagVal(val + self.f_offset)

@cython.final
cdef class LinearRamp(_RampFunctionBase):
    cdef readonly object start
    cdef readonly object end
    cdef double f_start
    cdef double f_end
    cdef double f_inv_length
    def __init__(self, start, end):
        self.start = start
        self.end = end

    cdef eval_end(self, length, oldval):
        return self.end

    cdef spline_segments(self, double length, double oldval):
        self.f_inv_length = 1 / length
        return ()

    @cython.cdivision(True)
    cdef int set_runtime_params(self, unsigned age) except -1:
        self.f_start = get_value_f64(self.start, age)
        self.f_end = get_value_f64(self.end, age)

    @cython.cdivision(True)
    cdef TagVal runtime_eval(self, double t) noexcept:
        t = t * self.f_inv_length
        return TagVal(self.f_start * (1 - t) + self.f_end * t)
