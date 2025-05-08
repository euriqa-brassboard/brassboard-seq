/*************************************************************************
 *   Copyright (c) 2025 - 2025 Yichao Yu <yyc1992@gmail.com>             *
 *                                                                       *
 *   This library is free software; you can redistribute it and/or       *
 *   modify it under the terms of the GNU Lesser General Public          *
 *   License as published by the Free Software Foundation; either        *
 *   version 3.0 of the License, or (at your option) any later version.  *
 *                                                                       *
 *   This library is distributed in the hope that it will be useful,     *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of      *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU    *
 *   Lesser General Public License for more details.                     *
 *                                                                       *
 *   You should have received a copy of the GNU Lesser General Public    *
 *   License along with this library. If not,                            *
 *   see <http://www.gnu.org/licenses/>.                                 *
 *************************************************************************/

#ifndef BRASSBOARD_SEQ_TEST_UTILS_H
#define BRASSBOARD_SEQ_TEST_UTILS_H

#include "src/utils.h"

using namespace brassboard_seq;

template<str_literal name> static auto new_object()
{
    static py::ptr type = py::ptr(&PyType_Type)(py::new_str(name), py::new_tuple(),
                                                py::new_dict()).rel();
    return type();
}

static auto value_pair_list(auto &values)
{
    return py::new_nlist(values.size(), [&] (int i) { return py::ptr(values[i].first); });
}

template<typename T> static inline auto to_py(T &&v);

template<typename> struct py_convert {
    static inline auto convert(bool b)
    {
        return py::new_bool(b);
    }
    static inline auto convert(std::integral auto i)
    {
        return py::new_int(i);
    }
    static inline auto convert(std::floating_point auto f)
    {
        return py::new_float(f);
    }
    template<typename T1, typename T2>
    static inline auto convert(const std::pair<T1,T2> &p)
    {
        return py::new_tuple(to_py(p.first), to_py(p.second));
    }
    template<typename T>
    static inline auto convert(const std::vector<T> &v)
    {
        return py::new_nlist(v.size(), [&] (int i) { return to_py(v[i]); });
    }
    template<typename K, typename V>
    static inline auto convert(const std::map<K,V> &m)
    {
        auto res = py::new_dict();
        for (auto [k, v]: m)
            res.set(to_py(k), to_py(v));
        return res;
    }
};

template<typename T, size_t N> struct py_convert<T[N]> {
    static inline auto convert(const auto *v)
    {
        return py::new_nlist(N, [&] (int i) { return to_py(v[i]); });
    }
};

template<typename T> static inline auto to_py(T &&v)
{
    return py_convert<std::remove_cvref_t<T>>::convert(std::forward<T>(v));
}

#endif
