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
#include "src/artiq_backend.h"

using namespace brassboard_seq;
using namespace brassboard_seq::artiq_backend;

template<>
struct py_convert<UrukulBus> {
    static auto convert(const UrukulBus &bus)
    {
        auto self = new_object<"UrukulBus">();
        self.set_attr("channel", to_py(bus.channel));
        self.set_attr("addr_target", to_py(bus.addr_target));
        self.set_attr("data_target", to_py(bus.data_target));
        self.set_attr("io_update_target", to_py(bus.io_update_target));
        self.set_attr("ref_period_mu", to_py(bus.ref_period_mu));
        return self;
    }
};

template<>
struct py_convert<DDSChannel> {
    static auto convert(const DDSChannel &dds)
    {
        auto self = new_object<"DDSChannel">();
        self.set_attr("ftw_per_hz", to_py(dds.ftw_per_hz));
        self.set_attr("bus_id", to_py(dds.bus_id));
        self.set_attr("chip_select", to_py(dds.chip_select));
        self.set_attr("delay", to_py(dds.delay));
        return self;
    }
};

template<>
struct py_convert<TTLChannel> {
    static auto convert(const TTLChannel &ttl)
    {
        auto self = new_object<"TTLChannel">();
        self.set_attr("target", to_py(ttl.target));
        self.set_attr("iscounter", to_py(ttl.iscounter));
        self.set_attr("delay", to_py(ttl.delay));
        return self;
    }
};

template<>
struct py_convert<ChannelType> {
    static auto convert(ChannelType type)
    {
        switch (type) {
        default:
        case DDSFreq: return "ddsfreq"_py;
        case DDSAmp: return "ddsamp"_py;
        case DDSPhase: return "ddsphase"_py;
        case TTLOut: return "ttl"_py;
        case CounterEnable: return "counter"_py;
        }
    }
};

template<>
struct py_convert<ChannelsInfo> {
    static auto convert(const ChannelsInfo &info)
    {
        auto self = new_object<"ChannelsInfo">();
        self.set_attr("urukul_busses", to_py(info.urukul_busses));
        self.set_attr("ttlchns", to_py(info.ttlchns));
        self.set_attr("ddschns", to_py(info.ddschns));
        self.set_attr("bus_chn_map", to_py(info.bus_chn_map));
        self.set_attr("ttl_chn_map", to_py(info.ttl_chn_map));
        self.set_attr("dds_param_chn_map", to_py(info.dds_param_chn_map));
        return self;
    }
};

template<>
struct py_convert<ArtiqAction> {
    static auto convert(const ArtiqAction &action)
    {
        auto self = new_object<"ArtiqAction">();
        self.set_attr("type", to_py(action.type));
        self.set_attr("cond", to_py(action.cond));
        self.set_attr("exact_time", to_py(action.exact_time));
        self.set_attr("chn_idx", to_py(action.chn_idx));
        self.set_attr("tid", to_py(action.tid));
        self.set_attr("time_mu", to_py(action.time_mu));
        self.set_attr("value", to_py(action.value));
        self.set_attr("aid", to_py(action.aid));
        self.set_attr("reloc_id", to_py(action.reloc_id));
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

template<>
struct py_convert<StartTrigger> {
    static auto convert(const StartTrigger &trig)
    {
        auto self = new_object<"StartTrigger">();
        self.set_attr("target", to_py(trig.target));
        self.set_attr("min_time_mu", to_py(trig.min_time_mu));
        self.set_attr("raising_edge", to_py(trig.raising_edge));
        self.set_attr("time_mu", to_py(trig.time_mu));
        return self;
    }
};

static PyModuleDef artiq_module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "brassboard_seq_artiq_backend_utils",
    .m_size = -1,
    .m_methods = (
        py::meth_table<
        py::meth_o<"get_channel_info",[] (auto, py::ptr<ArtiqBackend> ab) {
            return to_py(ab->data()->channels);
        }>,
        py::meth_o<"get_compiled_info",[] (auto, py::ptr<ArtiqBackend> ab) {
            auto &abd = *ab->data();
            auto self = new_object<"CompiledInfo">();
            self.set_attr("all_actions", to_py(abd.all_actions));
            self.set_attr("bool_values", value_pair_list(abd.bool_values));
            self.set_attr("float_values", value_pair_list(abd.float_values));
            self.set_attr("relocations", to_py(abd.relocations));
            return self;
        }>,
        py::meth_o<"get_start_trigger",[] (auto, py::ptr<ArtiqBackend> ab) {
            return to_py(ab->data()->start_triggers);
        }>,
        py::meth_fast<"add_start_trigger",[] (auto, PyObject *const *args,
                                              Py_ssize_t nargs) {
            py::check_num_arg("add_start_trigger", nargs, 5, 5);
            py::ptr<ArtiqBackend> ab = args[0];
            auto tgt = py::ptr(args[1]).template as_int<uint32_t>();
            auto time = py::ptr(args[2]).template as_int<int64_t>();
            auto min_time = py::ptr(args[3]).as_int();
            auto raising_edge = py::ptr(args[4]).as_bool();
            ab->data()->add_start_trigger_ttl(tgt, time, min_time, raising_edge);
        }>>),
};

PY_MODINIT(brassboard_seq_artiq_backend_utils, artiq_module)
{
}
