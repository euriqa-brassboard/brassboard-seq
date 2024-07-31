# cython: language_level=3

# Do not use relative import since it messes up cython file name tracking
from brassboard_seq.rtval cimport get_value, ifelse, is_rtval

cimport cython
from cpython cimport PyErr_Format, Py_LT, Py_GT

cdef np # hide import
import numpy as np
cimport numpy as cnpy
cnpy._import_array()

from libc cimport math as cmath

@cython.no_gc
@cython.final
cdef class Action:
    def __init__(self):
        PyErr_Format(TypeError, "Action cannot be created directly")

    def __richcmp__(self, other, int op):
        # For sorting actions according to their times.
        tid1 = self.tid
        if type(other) is not Action:
            # Action is a final type so we can use direct type comparison
            return NotImplemented
        tid2 = (<Action>other).tid
        if op == Py_LT:
            return tid1 < tid2
        elif op == Py_GT:
            return tid1 > tid2
        return NotImplemented

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

cdef class RampFunction:
    def __init__(self, *, **params):
        self.params = params
        self._eval = getattr(type(self), 'eval')
        try:
            self._spline_segments = getattr(type(self), 'spline_segments')
        except AttributeError:
            pass

cdef _cubic_spline_eval(SeqCubicSpline self, t, length, oldval):
    t = t / length
    if self.compile_mode:
        orders = <tuple>self._spline_segments
        return orders[0] + (orders[1] + (orders[2] + orders[3] * t) * t) * t
    return self.order0 + (self.order1 + (self.order2 + self.order3 * t) * t) * t

cdef cubic_spline_eval = _cubic_spline_eval

@cython.final
cdef class SeqCubicSpline:
    def __init__(self, order0, order1=0.0, order2=0.0, order3=0.0):
        self._spline_segments = (order0, order1, order2, order3)
        self._eval = cubic_spline_eval

cdef ramp_eval(RampFunction self, t, length, oldval):
    return self._eval(self, t, length, oldval)

cdef int ramp_set_compile_params(RampFunction self) except -1:
    if type(self) is SeqCubicSpline:
        (<SeqCubicSpline>self).compile_mode = True
        return 0
    for (name, value) in self.params.items():
        setattr(self, name, value)

cdef int ramp_set_runtime_params(RampFunction self, unsigned age) except -1:
    if type(self) is SeqCubicSpline:
        sp = <SeqCubicSpline>self
        orders = <tuple>sp._spline_segments
        sp.order0 = get_value(orders[0], age)
        sp.order1 = get_value(orders[1], age)
        sp.order2 = get_value(orders[2], age)
        sp.order3 = get_value(orders[3], age)
        sp.compile_mode = False
        return 0
    for (name, value) in self.params.items():
        setattr(self, name, get_value(value, age))

@cython.final
cdef class RampBuffer:
    def __init__(self):
        PyErr_Format(TypeError, "RampBuffer cannot be created directly")

cdef RampBuffer new_ramp_buffer():
    buff = <RampBuffer>RampBuffer.__new__(RampBuffer)
    return buff

cdef double *rampbuffer_alloc_input(_self, int size) except NULL:
    cdef RampBuffer self = <RampBuffer>_self
    cdef cnpy.npy_intp dims[1]
    cdef cnpy.PyArray_Dims pydims
    cdef cnpy.ndarray buff
    dims[0] = size
    if self.input_buff is None:
        buff = <cnpy.ndarray>cnpy.PyArray_EMPTY(1, dims, cnpy.NPY_DOUBLE, 0)
        self.input_buff = buff
    else:
        buff = <cnpy.ndarray>self.input_buff
        pydims.ptr = dims
        pydims.len = 1
        cnpy.PyArray_Resize(buff, &pydims, 0, cnpy.NPY_CORDER)
    return <double*>cnpy.PyArray_DATA(buff)

cdef double *rampbuffer_eval(_self, _func, length, oldval) except NULL:
    cdef RampBuffer self = <RampBuffer>_self
    func = <RampFunction?>_func
    buff = <cnpy.ndarray?>ramp_eval(func, self.input_buff, length, oldval)
    if buff.ndim != 1 or buff.size != len(self.input_buff):
        PyErr_Format(ValueError, "Ramp result dimension mismatch")
    if cnpy.PyArray_TYPE(buff) != cnpy.NPY_DOUBLE:
        buff = <cnpy.ndarray>cnpy.PyArray_Cast(buff, cnpy.NPY_DOUBLE)
    self.output_buff = buff
    return <double*>cnpy.PyArray_DATA(buff)

# These can be implemented in python code but are provided here
# to be slightly more efficient.
cdef np_cos = np.cos
cdef m_pi = cmath.pi
cdef m_2pi = cmath.pi * 2
cdef _blackman_eval(Blackman self, t, length, oldval):
    if not is_rtval(length) and length == 0:
        val = t * 0
    else:
        theta = t * (m_2pi / length) - m_pi
        cost = np_cos(theta)
        val = self.amp * (0.34 + cost * (0.5 + 0.16 * cost))
        val = ifelse(length == 0, t * 0, val)
    return val + self.offset
cdef blackman_eval = _blackman_eval

@cython.final
cdef class Blackman(RampFunction):
    cdef public object amp
    cdef public object offset
    def __init__(self, amp, offset=0):
        self.params = {'amp': amp, 'offset': offset}
        self._eval = blackman_eval

cdef _linear_eval(LinearRamp self, t, length, oldval):
    t = t / length
    return self.start * (1 - t) + self.end * t
cdef linear_eval = _linear_eval

cdef _linear_spline_segments(self, length, oldval):
    return ()
cdef linear_spline_segments = _linear_spline_segments

@cython.final
cdef class LinearRamp(RampFunction):
    cdef public object start
    cdef public object end
    def __init__(self, start, end):
        self.params = {'start': start, 'end': end}
        self._eval = linear_eval
        self._spline_segments = linear_spline_segments
