# cython: language_level=3

cimport cython

from cpython cimport PyErr_Format, PyObject, PyDict_GetItemWithError, PyDictProxy_New, PyTuple_GET_SIZE, PyDict_Size

cdef deepcopy_dict(v):
    if not isinstance(v, dict):
        return v
    cdef dict oldd = <dict>v
    cdef dict newd = {}
    for k, v in oldd.items():
        newd[k] = deepcopy_dict(v)
    return newd

cdef int merge_dict_into(dict tgt, dict src, bint ovr) except -1:
    cdef PyObject *oldvp
    for k, v in src.items():
        is_dict = isinstance(v, dict)
        oldvp = PyDict_GetItemWithError(tgt, k)
        if oldvp != NULL:
            oldv = <object>oldvp
            was_dict = isinstance(oldv, dict)
            if was_dict and not is_dict:
                PyErr_Format(TypeError, "Cannot override parameter pack as value")
            if not was_dict and is_dict:
                PyErr_Format(TypeError, "Cannot override value as parameter pack")
            if is_dict:
                merge_dict_into(<dict>oldv, <dict>v, ovr)
            elif ovr:
                tgt[k] = v
            continue
        if is_dict:
            tgt[k] = deepcopy_dict(v)
        else:
            tgt[k] = v
    return 0

cdef dict ensure_visited(ParamPack self):
    fieldname = self.fieldname
    self_visited = self.visited
    cdef PyObject *visitedp = PyDict_GetItemWithError(self_visited, fieldname)
    if visitedp == NULL:
        visited = {}
        self_visited[fieldname] = visited
        return visited
    return <dict>visitedp

cdef dict ensure_dict(ParamPack self):
    fieldname = self.fieldname
    values = self.values
    cdef PyObject *fieldp = PyDict_GetItemWithError(values, fieldname)
    if fieldp != NULL:
        field = <object>fieldp
        if isinstance(field, dict):
            return <dict>field
        PyErr_Format(TypeError, "Cannot access value as parameter pack.")
    field = {}
    values[fieldname] = field
    return <dict>field

cdef get_value(ParamPack self):
    fieldname = self.fieldname
    values = self.values
    cdef PyObject *fieldp = PyDict_GetItemWithError(values, fieldname)
    if fieldp == NULL:
        PyErr_Format(KeyError, "Value is not assigned")
    field = <object>fieldp
    if isinstance(field, dict):
        PyErr_Format(TypeError, "Cannot get parameter pack as value")
    self.visited[fieldname] = True
    return field

cdef get_value_default(ParamPack self, default_value):
    assert not isinstance(default_value, dict)
    fieldname = self.fieldname
    values = self.values
    cdef PyObject *fieldp = PyDict_GetItemWithError(values, fieldname)
    if fieldp != NULL:
        field = <object>fieldp
        if isinstance(field, dict):
            PyErr_Format(TypeError, "Cannot get parameter pack as value")
        return field
    values[fieldname] = default_value
    self.visited[fieldname] = True
    return default_value

@cython.final
cdef class ParamPack:
    def __init__(self, *args, **kwargs):
        self.values = {}
        self.visited = {}
        self.fieldname = 'root'
        nargs = PyTuple_GET_SIZE(args)
        nkws = PyDict_Size(kwargs)
        if nkws == 0 and nargs == 0:
            return
        self_values = ensure_dict(self)
        for arg in args:
            if not isinstance(arg, dict):
                PyErr_Format(TypeError,
                             "Cannot use value as default value for parameter pack")
            merge_dict_into(self_values, <dict>arg, False)
        if nkws != 0:
            merge_dict_into(self_values, kwargs, False)

    def __getattr__(self, str name):
        if name.startswith('_'):
            # IPython likes to poke the objects for various properties
            # when trying to show them. Stop all of these by forbidden
            # all attributes that starts with underscore.
            PyErr_Format(AttributeError,
                         "'ParamPack' object has no attribute '%U'", <PyObject*>name)
        return new_param_pack(ensure_dict(self), ensure_visited(self), name)

    def __setattr__(self, str name, value):
        if name.startswith('_'):
            # To be consistent with __getattr__
            PyErr_Format(AttributeError,
                         "'ParamPack' object has no attribute '%U'", <PyObject*>name)
        self_values = ensure_dict(self)
        cdef PyObject *oldvaluep = PyDict_GetItemWithError(self_values, name)
        if oldvaluep != NULL:
            oldvalue = <object>oldvaluep
            was_dict = isinstance(oldvalue, dict)
            is_dict = isinstance(value, dict)
            if was_dict and not is_dict:
                PyErr_Format(TypeError, "Cannot override parameter pack as value")
            if not was_dict and is_dict:
                PyErr_Format(TypeError, "Cannot override value as parameter pack")
            if is_dict:
                merge_dict_into(<dict>oldvalue, <dict>value, True)
            else:
                self_values[name] = value
        else:
            self_values[name] = deepcopy_dict(value)

    def __call__(self, *args, **kwargs):
        # Supported syntax
        # () -> get value without default
        # (value) -> get value with default
        # (*dicts, **kwargs) -> get parameter pack with default
        nargs = PyTuple_GET_SIZE(args)
        nkws = PyDict_Size(kwargs)
        if nkws == 0 and nargs == 0:
            return get_value(self)
        if nkws == 0 and nargs == 1:
            arg0 = args[0]
            if not isinstance(arg0, dict):
                return get_value_default(self, arg0)
        self_values = ensure_dict(self)
        for arg in args:
            if not isinstance(arg, dict):
                PyErr_Format(TypeError,
                             "Cannot use value as default value for parameter pack")
            merge_dict_into(self_values, <dict>arg, False)
        if nkws != 0:
            merge_dict_into(self_values, kwargs, False)
        return self

    def __str__(self):
        fieldname = self.fieldname
        values = self.values
        cdef PyObject *fieldp = PyDict_GetItemWithError(values, fieldname)
        if fieldp == NULL:
            return '<Undefined>'
        field = <object>fieldp
        if not isinstance(field, dict):
            return str(field)
        try:
            import yaml
            return yaml.dump(field)
        except:
            return str(field)

    def __repr__(self):
        return str(self)

cpdef get_visited(ParamPack self):
    fieldname = self.fieldname
    visited = self.visited
    cdef PyObject *resp = PyDict_GetItemWithError(visited, fieldname)
    if resp != NULL:
        res = <object>resp
        if isinstance(res, dict):
            return PyDictProxy_New(res)
        return res
    if isinstance(self.values.get(fieldname), dict):
        res = {}
        visited[fieldname] = res
        return PyDictProxy_New(res)
    return False

cdef ParamPack new_param_pack(dict value, dict visited, str fieldname):
    self = <ParamPack>ParamPack.__new__(ParamPack)
    self.values = value
    self.visited = visited
    self.fieldname = fieldname
    return self

# Helper function for functions that takes an optional parameter pack
def get_param(param, /):
    if param is None:
        return new_param_pack({}, {}, 'root')
    return param
