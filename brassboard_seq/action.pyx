# cython: language_level=3

cimport cython
from cpython cimport PyErr_Format, Py_LT, Py_GT

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
