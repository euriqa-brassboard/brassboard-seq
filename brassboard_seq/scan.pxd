# cython: language_level=3

cdef class ParamPack:
    cdef dict values
    cdef dict visited
    cdef str fieldname

cdef ParamPack new_param_pack(dict value, dict visited, str fieldname)
