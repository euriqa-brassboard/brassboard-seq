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

cdef extern from "src/config.h" namespace "brassboard_seq::config":
    ctypedef class brassboard_seq._utils.Config [object _brassboard_seq_config_Config, check_size ignore]:
        cdef dict channel_alias
        cdef dict alias_cache
        cdef set supported_prefix

    cdef tuple translate_channel(Config self, str name)

cdef extern from *:
    # Cython doesn't seem to allow namespace in the object property
    # for the imported extension class
    """
    using _brassboard_seq_config_Config = brassboard_seq::config::Config;
    """
