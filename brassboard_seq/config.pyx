# cython: language_level=3

cimport cython
from cpython cimport PyErr_Format, PyObject, PyDict_GetItemWithError, PyTuple_GET_SIZE

cdef class Config:
    def __cinit__(self):
        self.channel_alias = {}
        self.alias_cache = {}
        self.supported_prefix = set()

    cpdef void add_supported_prefix(self, str prefix):
        self.supported_prefix.add(prefix)

    cpdef void add_channel_alias(self, str name, str target):
        if '/' in name:
            PyErr_Format(ValueError, 'Channel alias name may not contain "/"')
        self.alias_cache.clear()
        self.channel_alias[name] = tuple(target.split('/'))

    @cython.final
    cdef tuple _translate_channel(self, tuple path):
        alias_cache = self.alias_cache
        cdef PyObject *resolvedp = PyDict_GetItemWithError(alias_cache, path)
        if resolvedp != NULL:
            resolved = <object>resolvedp
            if resolved is None:
                name = '/'.join(path)
                PyErr_Format(ValueError, 'Channel alias loop detected: %U',
                             <PyObject*>name)
            return <tuple>resolved
        if PyTuple_GET_SIZE(path) > 10: # Hardcoded limit for loop detection
            name = '/'.join(path)
            PyErr_Format(ValueError, 'Channel alias loop detected: %U',
                         <PyObject*>name)
        cdef str prefix = path[0]
        if prefix in self.channel_alias:
            newpath = <tuple>(self.channel_alias[prefix]) + path[1:]
            alias_cache[path] = None
            resolved = self._translate_channel(newpath)
            alias_cache[path] = resolved
            return resolved
        if not (prefix in self.supported_prefix):
            name = '/'.join(path)
            PyErr_Format(ValueError, 'Unsupported channel name: %U',
                         <PyObject*>name)
        alias_cache[path] = path
        return path

    cpdef tuple translate_channel(self, str name):
        return self._translate_channel(tuple(name.split('/')))
