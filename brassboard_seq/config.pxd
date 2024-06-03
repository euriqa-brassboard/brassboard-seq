# cython: language_level=3

cdef class Config:
    cdef dict channel_alias
    cdef dict alias_cache
    cdef set supported_prefix

    cpdef void add_supported_prefix(self, str prefix)
    cpdef void add_channel_alias(self, str name, str target)
    cdef tuple _translate_channel(self, tuple path)
    cpdef tuple translate_channel(self, str name)
