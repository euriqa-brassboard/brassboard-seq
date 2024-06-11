# cython: language_level=3

# Do not use relative import since it messes up cython file name tracking
from brassboard_seq.rtval cimport get_value

cimport cython
from cpython cimport PyErr_Format, Py_LT, Py_GT

cimport numpy as cnpy

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
        name = 'Pulse' if self.is_pulse else 'Set'
        kws = ''.join(f', {name}={val}' for (name, val) in self.kws.items())
        cond = self.cond
        if cond is not True:
            cond_str = f', cond={cond}'
        else:
            cond_str = ''
        if self.exact_time:
            return f'{name}({self.value}{cond_str}, exact_time=True{kws})'
        return f'{name}({self.value}{cond_str}{kws})'

    def __repr__(self):
        return str(self)

cdef class RampFunction:
    def __init__(self, *, **params):
        self.params = params

    @cython.final
    cdef int set_compile_params(self) except -1:
        for (name, value) in self.params.items():
            setattr(self, name, value)

    @cython.final
    cdef int set_runtime_params(self, long long age) except -1:
        for (name, value) in self.params.items():
            setattr(self, name, get_value(value, age))

    cpdef eval(self, t, length, oldval):
        return 0.0

@cython.final
cdef class RampBuffer:
    def __init__(self):
        PyErr_Format(TypeError, "RampBuffer cannot be created directly")

    cdef double *alloc_input(self, int size) except NULL:
        cdef cnpy.npy_intp dims[1]
        dims[0] = size
        cdef cnpy.PyArray_Dims pydims
        pydims.ptr = dims
        pydims.len = 1
        cdef cnpy.ndarray buff
        if self.input_buff is None:
            buff = <cnpy.ndarray>cnpy.PyArray_EMPTY(1, dims, cnpy.NPY_DOUBLE, 0)
            self.input_buff = buff
        else:
            buff = <cnpy.ndarray>self.input_buff
            cnpy.PyArray_Resize(buff, &pydims, 0, cnpy.NPY_CORDER)
        return <double*>cnpy.PyArray_DATA(buff)

    cdef double *eval(self, RampFunction func, double length,
                      double oldval) except NULL:
        buff = <cnpy.ndarray?>func.eval(self.input_buff, length, oldval)
        if buff.ndim != 1 or buff.size != len(self.input_buff):
            PyErr_Format(ValueError, "Ramp result dimension mismatch")
        if cnpy.PyArray_TYPE(buff) != cnpy.NPY_DOUBLE:
            buff = <cnpy.ndarray>cnpy.PyArray_Cast(buff, cnpy.NPY_DOUBLE)
        self.output_buff = buff
        return <double*>cnpy.PyArray_DATA(buff)

cdef RampBuffer new_ramp_buffer():
    cnpy.import_array()
    buff = <RampBuffer>RampBuffer.__new__(RampBuffer)
    return buff

cdef double *rampbuffer_alloc_input(PyObject *_self, int size) noexcept:
    cdef RampBuffer self = <RampBuffer>_self
    self.exception = None
    try:
        return self.alloc_input(size)
    except BaseException as e:
        self.exception = e
        return NULL

cdef double *rampbuffer_eval(PyObject *_self, PyObject *_func,
                             double length, double oldval) noexcept:
    cdef RampBuffer self = <RampBuffer>_self
    self.exception = None
    try:
        return self.eval(<RampFunction?>_func, length, oldval)
    except BaseException as e:
        self.exception = e
        return NULL
