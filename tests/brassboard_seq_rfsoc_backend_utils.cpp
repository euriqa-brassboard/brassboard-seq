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

#include "test_utils.h"
#include "src/rfsoc_backend.h"

using namespace brassboard_seq;
using namespace brassboard_seq::rfsoc_backend;

template<>
struct py_convert<RFSOCAction> {
    static auto convert(const RFSOCAction &action)
    {
        auto self = new_object<"RFSOCAction">();
        self.set_attr("cond", to_py(action.cond));
        self.set_attr("isramp", to_py(action.isramp));
        if (action.isramp)
            self.set_attr("ramp", action.ramp);
        self.set_attr("sync", to_py(action.sync));
        self.set_attr("reloc_id", to_py(action.reloc_id));
        self.set_attr("aid", to_py(action.aid));
        self.set_attr("tid", to_py(action.tid));
        self.set_attr("is_end", to_py(action.is_end));
        self.set_attr("seq_time", to_py(action.seq_time));
        self.set_attr("float_value", to_py(action.float_value));
        self.set_attr("bool_value", to_py(action.bool_value));
        return self;
    }
};

template<>
struct py_convert<ToneChannel> {
    static auto convert(const ToneChannel &tone_chn)
    {
        auto self = new_object<"ToneChannel">();
        self.set_attr("chn", to_py(tone_chn.chn));
        self.set_attr("actions", to_py(tone_chn.actions));
        return self;
    }
};

template<>
struct py_convert<ToneParam> {
    static auto convert(ToneParam param)
    {
        return py::new_str(param_name(param));
    }
};

template<>
struct py_convert<ChannelInfo> {
    static auto convert(const ChannelInfo &info)
    {
        auto self = new_object<"ChannelInfo">();
        self.set_attr("channels", to_py(info.channels));
        self.set_attr("chn_map", to_py(info.chn_map));
        self.set_attr("dds_delay", to_py(info.dds_delay));
        return self;
    }
};

template<>
struct py_convert<Relocation> {
    static auto convert(const Relocation &reloc)
    {
        auto self = new_object<"Relocation">();
        self.set_attr("cond_idx", to_py(reloc.cond_idx));
        self.set_attr("time_idx", to_py(reloc.time_idx));
        self.set_attr("val_idx", to_py(reloc.val_idx));
        return self;
    }
};

static PyModuleDef rfsoc_module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "brassboard_seq_rfsoc_backend_utils",
    .m_size = -1,
    .m_methods = (
        py::meth_table<
        py::meth_o<"get_channel_info",[] (auto, py::ptr<RFSOCBackend> rb) {
            return to_py(rb->data()->channels);
        }>,
        py::meth_o<"get_compiled_info",[] (auto, py::ptr<RFSOCBackend> rb) {
            auto self = new_object<"CompiiledInfo">();
            auto &rbd = *rb->data();
            self.set_attr("bool_values", value_pair_list(rbd.bool_values));
            self.set_attr("float_values", value_pair_list(rbd.float_values));
            self.set_attr("relocations", to_py(rbd.relocations));
            return self;
        }>>),
};

PY_MODINIT(brassboard_seq_rfsoc_backend_utils, rfsoc_module)
{
}
