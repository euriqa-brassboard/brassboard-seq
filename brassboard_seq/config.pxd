# cython: language_level=3

cdef class Config:
    cdef dict channel_alias
    cdef dict alias_cache
    cdef set supported_prefix

cdef tuple translate_channel(Config self, str name)
