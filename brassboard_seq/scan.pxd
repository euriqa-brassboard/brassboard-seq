# cython: language_level=3

cdef class ParamPack:
    cdef dict values
    cdef dict visited
    cdef str fieldname

    cdef dict ensure_visited(self)
    cdef dict ensure_dict(self)
    cdef get_value(self)
    cdef get_value_default(self, default_value)

cdef ParamPack new_param_pack(dict value, dict visited, str fieldname)
