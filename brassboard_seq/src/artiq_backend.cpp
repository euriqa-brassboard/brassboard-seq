/*************************************************************************
 *   Copyright (c) 2024 - 2025 Yichao Yu <yyc1992@gmail.com>             *
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

#include "artiq_backend.h"

#include "event_time.h"

#include <algorithm>

#include <assert.h>

#include <numpy/arrayobject.h>

namespace brassboard_seq::artiq_backend {

using rtval::RuntimeValue;
using event_time::EventTime;

namespace {

struct ArtiqInfo {
    py::ptr<> env_get_device;
    py::ptr<> AD9910;
    py::ptr<> EdgeCounter;
    py::ptr<> TTLOut;

    int COUNTER_ENABLE;
    int COUNTER_DISABLE;
    int _AD9910_REG_PROFILE0;
    int URUKUL_CONFIG;
    int URUKUL_CONFIG_END;
    int URUKUL_SPIT_DDS_WR;
    int URUKUL_DEFAULT_PROFILE{0};
    int SPI_CONFIG_ADDR;
    int SPI_DATA_ADDR;

    ArtiqInfo()
    {
        auto env = py::import_module("artiq.language.environment");

        auto ad9910 = py::import_module("artiq.coredevice.ad9910");
        auto edge_counter = py::import_module("artiq.coredevice.edge_counter");
        auto spi2 = py::import_module("artiq.coredevice.spi2");
        auto ttl = py::import_module("artiq.coredevice.ttl");
        auto urukul = py::import_module("artiq.coredevice.urukul");

        auto sim_ad9910 = py::try_import_module("dax.sim.coredevice.ad9910");
        auto sim_edge_counter = py::try_import_module("dax.sim.coredevice.edge_counter");
        auto sim_ttl = py::try_import_module("dax.sim.coredevice.ttl");

        env_get_device = env.attr("HasEnvironment").attr("get_device").rel();
        auto find_dev = [&] (auto &&mod, auto &&sim_mod, const char *name) {
            auto dev = mod.attr(name);
            if (sim_mod)
                dev = py::new_tuple(std::move(dev), sim_mod.attr(name));
            return dev;
        };
        AD9910 = find_dev(ad9910, sim_ad9910, "AD9910").rel();
        EdgeCounter = find_dev(edge_counter, sim_edge_counter, "EdgeCounter").rel();
        TTLOut = find_dev(ttl, sim_ttl, "TTLOut").rel();
        COUNTER_ENABLE = (edge_counter.attr("CONFIG_COUNT_RISING").as_int() |
                          edge_counter.attr("CONFIG_RESET_TO_ZERO").as_int());
        COUNTER_DISABLE = (edge_counter.attr("CONFIG_SEND_COUNT_EVENT").as_int());
        _AD9910_REG_PROFILE0 = ad9910.attr("_AD9910_REG_PROFILE0").as_int();
        URUKUL_CONFIG = urukul.attr("SPI_CONFIG").as_int();
        URUKUL_CONFIG_END = URUKUL_CONFIG | spi2.attr("SPI_END").as_int();
        URUKUL_SPIT_DDS_WR = urukul.attr("SPIT_DDS_WR").as_int();
        if (auto profile = urukul.try_attr("DEFAULT_PROFILE"))
            URUKUL_DEFAULT_PROFILE = profile.as_int();
        SPI_DATA_ADDR = spi2.attr("SPI_DATA_ADDR").as_int();
        SPI_CONFIG_ADDR = spi2.attr("SPI_CONFIG_ADDR").as_int();
    }

    auto get_device(py::ptr<> sys, py::ptr<> name) const
    {
        py::ref<> unique;
        if (auto registry = sys.try_attr("registry"_py)) {
            // DAX support
            unique = registry.attr("get_unique_device_key"_py)(name);
        }
        else {
            unique.assign(name);
        }
        // Do not call the get_device function from DAX since
        // it assumes that the calling object will take ownership of the deivce.
        return env_get_device(sys, unique);
    }
};

static inline const ArtiqInfo &info()
{
    static const ArtiqInfo info;
    return info;
}

}

__attribute__((visibility("internal")))
inline uint32_t ChannelsInfo::dds_to_mu(ChannelType type, int dds_idx, double v) const
{
    if (type == DDSFreq) {
        return uint32_t(v * ddschns[dds_idx].ftw_per_hz + 0.5);
    }
    else if (type == DDSAmp) {
        return std::clamp(int(v * 0x3fff + 0.5), 0, 0x3fff);
    }
    else {
        assert(type == DDSPhase);
        return round<int32_t>(v * 0x10000) & 0xffff;
    }
}

__attribute__((visibility("internal")))
inline void ChannelsInfo::add_channel(py::ptr<> dev, int64_t delay, rtval_ptr rt_delay,
                                      int idx, py::tuple path)
{
    if (dev.isinstance(info().AD9910)) {
        if (path.size() > 3)
            config::raise_invalid_channel(path);
        auto path2 = path.get<py::str>(2);
        ChannelType dds_param_type;
        if (path2.compare_ascii("sw") == 0) {
            // Note that we currently do not treat this switch ttl channel
            // differently from any other ttl channels.
            // We may consider maintaining a relation between this ttl channel
            // and the urukul channel to make sure we don't reorder
            // any operations between the two.
            add_ttl_channel(idx, dev.attr("sw"_py).attr("target_o"_py).as_int(),
                            false, delay, rt_delay);
            return;
        }
        else if (path2.compare_ascii("freq") == 0) {
            dds_param_type = DDSFreq;
        }
        else if (path2.compare_ascii("amp") == 0) {
            dds_param_type = DDSAmp;
        }
        else if (path2.compare_ascii("phase") == 0) {
            dds_param_type = DDSPhase;
        }
        else {
            config::raise_invalid_channel(path);
        }
        auto bus = dev.attr("bus"_py);
        auto bus_channel = bus.attr("channel"_py).as_int();
        auto bus_id = find_bus_id(bus_channel);
        if (bus_id == -1) {
            // Here we assume that the CPLD (and it's io_update channel)
            // and the SPI bus has a one-to-one mapping.
            // This means that each DDS with the same bus shares
            // the same io_update channel and can only be programmed one at a time.
            auto io_tgt = dev.attr("cpld"_py).attr("io_update"_py)
                .attr("target_o"_py).as_int();
            bus_id = add_bus_channel(bus_channel, io_tgt,
                                     bus.attr("ref_period_mu"_py).as_int());
        }
        add_dds_param_channel(idx, bus_id, dev.attr("ftw_per_hz"_py).as_float(),
                              dev.attr("chip_select"_py).as_int(), dds_param_type,
                              delay, rt_delay);
    }
    else if (dev.isinstance(info().TTLOut)) {
        if (path.size() > 2)
            config::raise_invalid_channel(path);
        add_ttl_channel(idx, dev.attr("target_o"_py).as_int(), false, delay, rt_delay);
    }
    else if (dev.isinstance(info().EdgeCounter)) {
        if (path.size() > 2)
            config::raise_invalid_channel(path);
        add_ttl_channel(idx, dev.attr("channel"_py).as_int() << 8, true, delay, rt_delay);
    }
    else {
        py_throw_format(PyExc_ValueError, "Unsupported device: %S", dev);
    }
}

__attribute__((visibility("internal")))
inline int ChannelsInfo::add_bus_channel(int bus_channel, uint32_t io_update_target,
                                         uint8_t ref_period_mu)
{
    auto bus_id = (int)urukul_busses.size();
    urukul_busses.push_back({
            uint32_t(bus_channel),
            uint32_t((bus_channel << 8) | info().SPI_CONFIG_ADDR),
            uint32_t((bus_channel << 8) | info().SPI_DATA_ADDR),
            // Here we assume that the CPLD (and it's io_update channel)
            // and the SPI bus has a one-to-one mapping.
            // This means that each DDS with the same bus shares
            // the same io_update channel and can only be programmed one at a time.
            io_update_target,
            ref_period_mu,
        });
    bus_chn_map[bus_channel] = bus_id;
    return bus_id;
}

__attribute__((visibility("internal")))
inline void ChannelsInfo::add_ttl_channel(int seqchn, uint32_t target, bool iscounter,
                                          int64_t delay, rtval_ptr rt_delay)
{
    assert(ttl_chn_map.count(seqchn) == 0);
    auto ttl_id = (int)ttlchns.size();
    ttlchns.push_back({target, iscounter, delay, rt_delay});
    ttl_chn_map[seqchn] = ttl_id;
}

__attribute__((visibility("internal")))
inline int ChannelsInfo::get_dds_channel_id(uint32_t bus_id, double ftw_per_hz,
                                            uint8_t chip_select, int64_t delay,
                                            rtval_ptr rt_delay)
{
    std::pair<int,int> key{bus_id, chip_select};
    auto it = dds_chn_map.find(key);
    if (it != dds_chn_map.end())
        return it->second;
    auto dds_id = (int)ddschns.size();
    ddschns.push_back({ .ftw_per_hz = ftw_per_hz, .bus_id = bus_id,
            .chip_select = chip_select, .delay = delay, .rt_delay = rt_delay });
    dds_chn_map[key] = dds_id;
    return dds_id;
}

__attribute__((visibility("internal")))
inline void ChannelsInfo::add_dds_param_channel(int seqchn, uint32_t bus_id,
                                                double ftw_per_hz, uint8_t chip_select,
                                                ChannelType param, int64_t delay,
                                                rtval_ptr rt_delay)
{
    assert(dds_param_chn_map.count(seqchn) == 0);
    dds_param_chn_map[seqchn] = {get_dds_channel_id(bus_id, ftw_per_hz, chip_select,
                                                    delay, rt_delay), param};
}

__attribute__((visibility("internal")))
inline void ChannelsInfo::collect_channels(py::str prefix, py::ptr<> sys,
                                           py::ptr<seq::Seq> seq, py::dict device_delay)
{
    for (auto [idx, path]: py::list_iter<py::tuple>(seq->seqinfo->channel_paths)) {
        if (path.get<py::str>(0).compare(prefix) != 0)
            continue;
        if (path.size() < 2)
            config::raise_invalid_channel(path);
        auto devname = path.get<py::str>(1);
        int64_t delay = 0;
        rtval_ptr rt_delay;
        if (auto py_delay = device_delay.try_get(devname)) {
            rt_delay = py::cast<RuntimeValue>(py_delay);
            if (!rt_delay) {
                delay = py_delay.as_int<int64_t>();
            }
        }
        add_channel(info().get_device(sys, devname), delay, rt_delay, idx, path);
    }
    dds_chn_map.clear(); // Not needed after channel collection
}

__attribute__((visibility("internal")))
void UrukulBus::add_dds_action(RTIOGen &gen, DDSAction &action)
{
    auto div = info().URUKUL_SPIT_DDS_WR;
    auto ddschn = action.ddschn;
    bb_debug("add_dds_action: aid=%d, bus@%" PRId64 ", io_upd@%" PRId64 ", "
             "data1=%x, data2=%x, chn=%d, cs=%d\n",
             action.aid, last_bus_mu, last_io_update_mu,
             action.data1, action.data2, channel, ddschn->chip_select);

    auto config_and_write = [&] (uint32_t flags, uint32_t length,
                                 uint32_t data, int64_t lb_mu1, int64_t lb_mu2) {
        auto addr = (flags | ((length - 1) << 8) | ((div - 2) << 16) |
                     (uint32_t(ddschn->chip_select) << 24));
        uint16_t data_time_mu = uint16_t(((length + 1) * div + 1) * ref_period_mu);
        auto t1 = gen.add_action(addr_target, addr, action.aid, lb_mu1, lb_mu1, false);
        lb_mu2 = std::max(lb_mu2, t1 + ref_period_mu);
        auto t2 = gen.add_action(data_target, data, action.aid, lb_mu2, lb_mu2, false);
        return t2 + data_time_mu;
    };
    auto profile_reg = (info()._AD9910_REG_PROFILE0 +
                        info().URUKUL_DEFAULT_PROFILE);
    // We can't start the write safely before the previous io_update finishes
    // since it may abort the write. However, we can configure the SPI controller
    // before the io_update finishes since no signal should be sent to the DDS.
    auto t1 = config_and_write(info().URUKUL_CONFIG, 8, profile_reg << 24,
                               last_bus_mu, std::max(last_bus_mu, last_io_update_mu));
    auto t2 = config_and_write(info().URUKUL_CONFIG, 32, action.data1, t1, t1);
    auto t3 = config_and_write(info().URUKUL_CONFIG_END, 32, action.data2,
                               t2, t2);
    ddschn->data1 = action.data1;
    ddschn->data2 = action.data2;
    ddschn->had_output = true;
    last_bus_mu = t3;
}

__attribute__((visibility("internal")))
void UrukulBus::add_io_update(RTIOGen &gen, int64_t time_mu,
                              int aid, bool exact_time)
{
    // Round to the nearest 8 cycles.
    time_mu = (time_mu + coarse_time_mu / 2) & ~int64_t(coarse_time_mu - 1);
    bb_debug("add_io_update: aid=%d, bus@%" PRId64 ", io_upd@%" PRId64 ", "
             "time=%" PRId64 ", exact_time=%d, chn=%d\n",
             aid, last_bus_mu, last_io_update_mu, time_mu, exact_time, channel);
    auto t1 = gen.add_action(io_update_target, 1, aid, time_mu,
                             std::max(last_bus_mu, last_io_update_mu), exact_time);
    auto t2 = gen.add_action(io_update_target, 0, aid, t1 + coarse_time_mu,
                             t1 + coarse_time_mu, false);
    last_io_update_mu = t2;
}

__attribute__((visibility("internal")))
inline void UrukulBus::flush_output(RTIOGen &gen, int64_t time_mu, bool force)
{
    if (!dds_actions.empty())
        bb_debug("flush_dds: bus@%" PRId64 ", io_upd@%" PRId64 ", "
                 "time=%" PRId64 ", chn=%d, nactions=%zd, force=%d\n",
                 last_bus_mu, last_io_update_mu, time_mu, channel,
                 dds_actions.size(), (int)force);
    bool need_io_update = false;
    int aid = -1;
    int64_t update_time_mu = 0;
    while (!dds_actions.empty()) {
        auto &action = dds_actions.front();
        if (!force && action.time_mu + max_action_shift >= time_mu)
            break;
        add_dds_action(gen, action);
        if (action.exact_time) {
            add_io_update(gen, action.time_mu, action.aid, true);
        }
        else {
            need_io_update = true;
            aid = action.aid;
            update_time_mu = action.time_mu;
        }
        dds_actions.erase(dds_actions.begin());
    }
    if (need_io_update) {
        add_io_update(gen, update_time_mu, aid, false);
    }
}

static inline void update_dds_data(uint32_t &data1, uint32_t &data2,
                                   ChannelType type, uint32_t value)
{
    if (type == DDSFreq) {
        data2 = value;
    }
    else if (type == DDSAmp) {
        data1 = (data1 & 0xffff) | (value << 16);
    }
    else {
        assert(type == DDSPhase);
        data1 = (data1 & 0xffff0000) | value;
    }
}

__attribute__((visibility("internal")))
inline void UrukulBus::add_output(RTIOGen &gen, const ArtiqAction &action,
                                  DDSChannel &ddschn)
{
    bb_debug("add_dds: time=%" PRId64 ", exact_time=%d, chn=%d, nactions=%zd, "
             "type=%s, value=%x\n",
             action.time_mu, action.exact_time, channel, dds_actions.size(),
             (action.type == DDSFreq ? "ddsfreq" :
              (action.type == DDSAmp ? "ddsamp" : "ddsphase")), action.value);
    for (auto it = dds_actions.begin(), end = dds_actions.end(); it != end; ++it) {
        auto &dds_action = *it;
        if (dds_action.ddschn == &ddschn) {
            if (action.exact_time && dds_action.exact_time &&
                action.time_mu != dds_action.time_mu) {
                // Flush output and add a new one afterwards
                bb_debug("add_dds: found pending exact action at time=%" PRId64 ", "
                         "flusing\n", dds_action.time_mu);
                auto last = it + 1;
                for (auto it2 = dds_actions.begin(); it2 != last; ++it2) {
                    auto &dds_action2 = *it2;
                    add_dds_action(gen, dds_action2);
                    if (action.exact_time) {
                        add_io_update(gen, dds_action2.time_mu,
                                      dds_action2.aid, true);
                    }
                }
                dds_actions.erase(dds_actions.begin(), last);
                break;
            }
            update_dds_data(dds_action.data1, dds_action.data2, action.type, action.value);
            if (ddschn.had_output && dds_action.data1 == ddschn.data1 &&
                dds_action.data2 == ddschn.data2) {
                bb_debug("add_dds: new values the same as the old values "
                         "(data1=%x, data2=%x), erasing\n",
                         dds_action.data1, dds_action.data2);
                // No-op, pop the action
                dds_actions.erase(it);
                return;
            }
            // Data already updated, we are done if the new action is not exact time.
            if (action.exact_time) {
                bb_debug("add_dds: Moving pending non-exact time action to new time "
                         "(data1=%x, data2=%x)\n", dds_action.data1, dds_action.data2);
                // Move action to the end
                auto new_action = dds_action;
                new_action.exact_time = true;
                new_action.time_mu = action.time_mu;
                new_action.aid = action.aid;
                dds_actions.erase(it);
                dds_actions.push_back(new_action);
            }
            return;
        }
    }
    // New action
    uint32_t data1 = ddschn.data1;
    uint32_t data2 = ddschn.data2;
    update_dds_data(data1, data2, action.type, action.value);
    if (ddschn.had_output && data1 == ddschn.data1 && data2 == ddschn.data2) {
        bb_debug("add_dds: new values the same as the old values "
                 "(data1=%x, data2=%x), skipping\n", data1, data2);
        return;
    }
    bb_debug("add_dds: adding pending dds action (data1=%x, data2=%x), skipping\n",
             data1, data2);
    dds_actions.push_back({action.time_mu, data1, data2, action.exact_time,
            action.aid, &ddschn});
}

__attribute__((visibility("internal")))
inline void TTLChannel::flush_output(RTIOGen &gen, int64_t cur_time_mu,
                                     bool exact_time_only, bool force)
{
    if (new_val == cur_val)
        return;
    if (cur_time_mu <= time_mu + max_action_shift && !force)
        return;
    if (exact_time_only && !exact_time)
        return;
    bb_debug("flush_ttl: last_time@%" PRId64 ", "
             "time=%" PRId64 ", tgt=%d, cur_v=%d, new_v=%d, force=%d\n",
             last_time_mu, cur_time_mu, target, cur_val, new_val, (int)force);
    // Now do the output
    cur_val = new_val;
    last_time_mu = gen.add_action(target, new_val, aid, time_mu,
                                  last_time_mu, exact_time) + coarse_time_mu;
}

__attribute__((visibility("internal")))
inline uint8_t TTLChannel::ttl_to_mu(bool v) const
{
    if (!iscounter) {
        return v;
    }
    else if (v) {
        return info().COUNTER_ENABLE;
    }
    else {
        return info().COUNTER_DISABLE;
    }
}

__attribute__((visibility("internal")))
inline void TTLChannel::add_output(RTIOGen &gen, const ArtiqAction &action)
{
    uint8_t val = ttl_to_mu(action.value);

    bb_debug("add_ttl: time=%" PRId64 ", exact_time=%d, tgt=%d, "
             "iscounter=%d, value=%x\n", action.time_mu, action.exact_time, target,
             iscounter, action.value);

    // No need to output
    if (new_val == val) {
        bb_debug("add_ttl: value already queued, skipping\n");
        return;
    }

    // For counters, we need to maintain the on-off count since that determines
    // the number of results we produce. we'll therefore always do the output
    // immediately and not allowing any coalesce of values
    if (iscounter) {
        bb_debug("add_ttl: flush counter channel\n");
        assert(cur_val == new_val);
        cur_val = val;
        new_val = val;
        last_time_mu = gen.add_action(target, val, action.aid, action.time_mu,
                                      last_time_mu, action.exact_time) + coarse_time_mu;
        return;
    }

    // If we have two exact outputs we need to flush the previous one
    if (cur_val != new_val && exact_time && action.exact_time &&
        action.time_mu != time_mu) {
        bb_debug("add_ttl: flush pending exact time output\n");
        last_time_mu = gen.add_action(target, new_val, aid, time_mu,
                                      last_time_mu, true) + coarse_time_mu;
        cur_val = new_val;
    }

    // No current output
    if (cur_val == new_val) {
        bb_debug("add_ttl: queuing new output\n");
        new_val = val;
        exact_time = action.exact_time;
        time_mu = action.time_mu;
        aid = action.aid;
        return;
    }

    bb_debug("add_ttl: updating queued output to new value\n");
    if (action.exact_time) {
        time_mu = action.time_mu;
        exact_time = true;
    }
    new_val = val;
}

__attribute__((visibility("internal")))
inline void TimeChecker::clear()
{
    counts.clear();
    max_key = 0;
}

__attribute__((visibility("internal")))
inline bool TimeChecker::check_and_add_time(int64_t t_mu)
{
    auto t_course = t_mu / coarse_time_mu;
    if (t_course > max_key) {
        max_key = t_course;
        counts[t_course] = 1;
        return true;
    }
    auto &cnt = counts[t_course];
    if (cnt >= 7)
        return false;
    cnt += 1;
    return true;
}

__attribute__((visibility("internal")))
inline int64_t TimeChecker::find_time(int64_t lb_mu, int64_t t_mu, int64_t ub_mu)
{
    if (check_and_add_time(t_mu))
        return t_mu;
    int64_t offset = coarse_time_mu;
    while (true) {
        auto in_lb = t_mu - offset >= lb_mu;
        auto in_ub = t_mu + offset <= ub_mu;
        if (in_lb && check_and_add_time(t_mu - offset))
            return t_mu - offset;
        if (in_ub && check_and_add_time(t_mu + offset))
            return t_mu + offset;
        if (!in_lb && !in_ub)
            return INT64_MIN;
        offset += coarse_time_mu;
    }
}

struct __attribute__((visibility("internal"))) ArtiqBackend::Data::Indexers {
    ValueIndexer<bool> bool_values;
    ValueIndexer<double> float_values;
};

__attribute__((visibility("internal")))
auto ArtiqBackend::Output::alloc(int bseq_id, py::ref<> rtios, bool may_term,
                                 std::vector<int> next_cbseq) -> py::ref<Output>
{
    auto self = py::generic_alloc<Output>();
    self->bseq_id = bseq_id;
    self->may_term = may_term;
    call_constructor(&self->next_cbseq, std::move(next_cbseq));
    call_constructor(&self->actions);
    call_constructor(&self->start_values);
    call_constructor(&self->rtios, std::move(rtios));
    self->total_time_mu = 0;
    return self;
}
__attribute__((visibility("internal")))
PyTypeObject ArtiqBackend::Output::Type = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "brassboard_seq.artiq_backend.Output",
    .tp_basicsize = sizeof(Output),
    .tp_dealloc = py::tp_cxx_dealloc<true,Output>,
    .tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_HAVE_GC,
    .tp_traverse = py::tp_field_traverse<Output,&Output::rtios>,
    .tp_clear = py::tp_field_clear<Output,&Output::rtios>,
    .tp_members = (py::mem_table<
                   py::mem_def<"bseq_id",T_INT,&Output::bseq_id,READONLY>,
                   py::mem_def<"may_term",T_BOOL,&Output::may_term,READONLY>,
                   py::mem_def<"rtios",T_OBJECT_EX,&Output::rtios,READONLY>>),
    .tp_getset = (py::getset_table<
                  py::getset_def<"next",[] (py::ptr<Output> self) {
                      return to_py(self->next_cbseq);
                  }>,
                  py::getset_def<"total_time_mu",[] (py::ptr<Output> self) {
                      return to_py(self->total_time_mu);
                  }>>),
};

__attribute__((visibility("internal")))
inline ArtiqBackend::Data::Data(py::ptr<> sys, py::ref<> all_outputs, bool use_dma,
                                bool support_branch)
    : sys(sys.ref()),
      use_dma(use_dma),
      support_branch(support_branch),
      all_outputs(std::move(all_outputs))
{
}

__attribute__((visibility("internal")))
void ArtiqBackend::Data::process_bseq(py::ptr<SeqCompiler> comp, CompiledBasicSeq &cbseq,
                                      py::ptr<Output> output, Indexers &idr)
{
    auto &artiq_actions = output->actions;
    int bseq_id = cbseq.bseq_id;
    py::ptr bseq = comp->seq->basic_seqs.get<seq::BasicSeq>(bseq_id);
    py::list event_times = bseq->seqinfo->time_mgr->event_times;

    auto add_single_action = [&] (action::Action *action, ChannelType type, int chn_idx,
                                  int tid, py::ptr<> value, int cond_reloc,
                                  bool is_end) {
        ArtiqAction artiq_action;
        int aid = action->aid;
        artiq_action.type = type;
        artiq_action.cond = true;
        artiq_action.exact_time = action->exact_time;
        artiq_action.eval_status = false;
        artiq_action.chn_idx = chn_idx;
        artiq_action.tid = tid;
        artiq_action.is_end = is_end;
        artiq_action.aid = aid;

        bool needs_reloc = cond_reloc != -1;
        Relocation reloc{cond_reloc, -1, -1};

        auto event_time = event_times.get<EventTime>(tid);
        if (event_time->data.is_static()) {
            rtval_ptr rt_delay;
            int64_t delay;
            if (type == DDSFreq || type == DDSAmp || type == DDSPhase) {
                auto &ddschn = channels.ddschns[chn_idx];
                delay = ddschn.delay;
                rt_delay = ddschn.rt_delay;
            }
            else {
                auto &ttlchn = channels.ttlchns[chn_idx];
                delay = ttlchn.delay;
                rt_delay = ttlchn.rt_delay;
            }
            if (rt_delay) {
                // We need to fill in the time at runtime
                // since we don't know the delay value...
                needs_reloc = true;
                reloc.time_idx = tid;
            }
            else {
                artiq_action.time_mu =
                    seq_time_to_mu(event_time->data._get_static() + delay);
            }
        }
        else {
            needs_reloc = true;
            reloc.time_idx = tid;
        }
        if (rtval::is_rtval(value)) {
            needs_reloc = true;
            if (type == DDSFreq || type == DDSAmp || type == DDSPhase) {
                reloc.val_idx = idr.float_values.get_id(value);
            }
            else if (type == CounterEnable) {
                // We aren't really relying on this in the backend
                // but requiring this makes it easier to infer the number of
                // results generated from a sequence.
                bb_throw_format(PyExc_ValueError, action_key(aid),
                                "Counter value must be static.");
            }
            else {
                reloc.val_idx = idr.bool_values.get_id(value);
            }
        }
        else if (type == DDSFreq || type == DDSAmp || type == DDSPhase) {
            artiq_action.value = channels.dds_to_mu(type, chn_idx,
                                                    value.as_float(action_key(aid)));
        }
        else {
            artiq_action.value = value.as_bool(action_key(aid));
        }
        if (needs_reloc) {
            artiq_action.reloc_id = (int)relocations.size();
            relocations.push_back(reloc);
        }
        else {
            artiq_action.reloc_id = -1;
        }
        artiq_actions.push_back(artiq_action);
    };
    auto add_artiq_action = [&] (action::Action *action, ChannelType type, int chn_idx) {
        py::ptr cond = action->cond;
        int cond_reloc = -1;
        if (rtval::is_rtval(cond)) {
            cond_reloc = idr.bool_values.get_id(cond);
            assume(cond_reloc >= 0);
        }
        else {
            assert(cond == Py_True);
        }
        add_single_action(action, type, chn_idx, action->tid, action->value,
                          cond_reloc, false);
        if (action->is_pulse) {
            add_single_action(action, type, chn_idx, action->end_tid,
                              action->end_val, cond_reloc, true);
        }
    };

    // FIXME:
    // We use a different semantics for start value for the first bseq compared to others.
    // For the first bseq, we assume the channel is unknown but we only initialize
    // the channel when we use it. For the rest of the bseq, we assume the channel
    // already has the value recorded in start value by previous bseq.
    // There is in priciple a problem that if a channel is not used in the first bseq
    // then the first use of it in later bseq would not initialize it if the value
    // we want to set it to matches the start value.
    bool is_entry = &cbseq == &comp->basic_cseqs[0];
    for (auto [chn, ttl_idx]: channels.ttl_chn_map) {
        auto ttl_chn_info = channels.ttlchns[ttl_idx];
        auto type = ttl_chn_info.iscounter ? CounterEnable : TTLOut;
        auto chn_action = cbseq.chn_actions[chn];
        if (!is_entry) {
            StartValue start_value{ .type = type, .chn_idx = ttl_idx };
            if (rtval::is_rtval(chn_action->start_value)) {
                start_value.val_id = idr.bool_values.get_id(chn_action->start_value.get());
            }
            else {
                start_value.val_id = -1;
                start_value.value = chn_action->start_value.as_bool();
            }
            output->start_values.push_back(start_value);
        }
        for (auto action: chn_action->actions) {
            if (action->kws)
                bb_throw_format(PyExc_ValueError, action_key(action->aid),
                                "Invalid output keyword argument %S", action->kws);
            if (action::isramp(action->value))
                bb_throw_format(PyExc_ValueError, action_key(action->aid),
                                "TTL Channel cannot be ramped");
            if (action->cond == Py_False)
                continue;
            add_artiq_action(action, type, ttl_idx);
        }
    }

    for (auto [chn, value]: channels.dds_param_chn_map) {
        auto [dds_idx, type] = value;
        auto chn_action = cbseq.chn_actions[chn];
        if (!is_entry) {
            StartValue start_value{ .type = type, .chn_idx = dds_idx };
            if (rtval::is_rtval(chn_action->start_value)) {
                start_value.val_id =
                    idr.float_values.get_id(chn_action->start_value.get());
            }
            else {
                start_value.val_id = -1;
                start_value.value =
                    channels.dds_to_mu(type, dds_idx, chn_action->start_value.as_float());
            }
            output->start_values.push_back(start_value);
        }
        for (auto action: chn_action->actions) {
            if (action->kws)
                bb_throw_format(PyExc_ValueError, action_key(action->aid),
                                "Invalid output keyword argument %S", action->kws);
            if (action::isramp(action->value))
                bb_throw_format(PyExc_ValueError, action_key(action->aid),
                                "DDS Channel cannot be ramped");
            if (action->cond == Py_False)
                continue;
            add_artiq_action(action, type, dds_idx);
        }
    }
}

__attribute__((visibility("internal")))
void ArtiqBackend::Data::finalize(py::ptr<SeqCompiler> comp)
{
    if (comp->basic_cseqs.size() != 1 && !support_branch)
        py_throw_format(PyExc_ValueError,
                        "Artiq backend not initialized for branch support");
    channels.collect_channels(prefix, sys, comp->seq, device_delay);
    if (support_branch) {
        assert(all_outputs.size() == 0);
        auto new_rtios = [&] () -> py::ref<> {
            if (use_dma)
                return py::new_bytearray();
            npy_intp sz = 0;
            return py::ref(PyArray_SimpleNew(1, &sz, NPY_INT32));
        };
        for (auto &cbseq: comp->basic_cseqs) {
            all_outputs.append(Output::alloc(cbseq.bseq_id, new_rtios(),
                                             cbseq.may_term, cbseq.next_bseq));
        }
    }

    Indexers idr;
    for (auto [cbseq_id, output]: py::list_iter(all_outputs))
        process_bseq(comp, comp->basic_cseqs[cbseq_id], output, idr);

    bool_values = std::move(idr.bool_values.values);
    float_values = std::move(idr.float_values.values);
}

static inline int64_t convert_device_delay(double fdelay)
{
    if (fdelay < 0) {
        py_throw_format(PyExc_ValueError, "Device time offset %S cannot be negative.",
                        to_py(fdelay));
    }
    else if (fdelay > 0.1) {
        py_throw_format(PyExc_ValueError,
                        "Device time offset %S cannot be more than 100ms.", to_py(fdelay));
    }
    return int64_t(fdelay * event_time::time_scale + 0.5);
}

__attribute__((visibility("internal")))
void ArtiqBackend::Data::reloc_action(const ArtiqAction &action,
                                      const std::vector<int64_t> &time_values) const
{
    auto reloc = relocations[action.reloc_id];
    if (reloc.cond_idx != -1)
        action.cond = bool_values[reloc.cond_idx].second;
    // No need to do anything else if we hit a disabled action.
    if (!action.cond)
        return;
    auto type = action.type;
    auto chn_idx = action.chn_idx;
    if (reloc.time_idx != -1) {
        int64_t delay;
        if (type == DDSFreq || type == DDSAmp || type == DDSPhase) {
            delay = channels.ddschns[chn_idx].delay;
        }
        else {
            delay = channels.ttlchns[chn_idx].delay;
        }
        action.time_mu = seq_time_to_mu(time_values[reloc.time_idx] + delay);
    }
    if (reloc.val_idx != -1) {
        if (type == DDSFreq || type == DDSAmp || type == DDSPhase) {
            action.value = channels.dds_to_mu(type, chn_idx,
                                              float_values[reloc.val_idx].second);
        }
        else {
            action.value = bool_values[reloc.val_idx].second;
        }
    }
}

__attribute__((visibility("internal")))
int64_t RTIOGen::add_action(uint32_t target, uint32_t value, int aid,
                            int64_t request_time_mu, int64_t lb_mu,
                            bool exact_time)
{
    if (exact_time) {
        if (request_time_mu < lb_mu) {
            bb_throw_format(PyExc_ValueError, action_key(aid),
                            "Exact time output cannot satisfy lower time bound");
        }
        else if (!time_checker.check_and_add_time(request_time_mu)) {
            bb_throw_format(PyExc_ValueError, action_key(aid),
                            "Too many outputs at the same time");
        }
        actions.push_back({target, value, request_time_mu});
        return request_time_mu;
    }
    // hard code a 10 us bound.
    constexpr int64_t max_offset = 10000;
    int64_t ub_mu = request_time_mu + max_offset;
    if (ub_mu < lb_mu)
        bb_throw_format(PyExc_ValueError, action_key(aid),
                        "Cannot find appropriate output time within bound");
    if (request_time_mu < lb_mu) {
        request_time_mu = lb_mu;
    }
    else {
        lb_mu = std::max(lb_mu, request_time_mu - max_offset);
    }
    auto time_mu = time_checker.find_time(lb_mu, request_time_mu, ub_mu);
    if (time_mu == INT64_MIN)
        bb_throw_format(PyExc_ValueError, action_key(aid),
                        "Too many outputs at the same time");
    actions.push_back({target, value, time_mu});
    return time_mu;
}

__attribute__((visibility("internal")))
void ArtiqBackend::Data::generate_bseq(py::ptr<SeqCompiler> comp,
                                       CompiledBasicSeq &cbseq, py::ptr<Output> output)
{
    int bseq_id = output->bseq_id;
    py::ptr bseq = comp->seq->basic_seqs.get<seq::BasicSeq>(bseq_id);
    auto &time_values = bseq->seqinfo->time_mgr->time_values;

    auto &artiq_actions = output->actions;
    if (artiq_actions.size() == 1) {
        auto &a = artiq_actions[0];
        if (a.reloc_id >= 0) {
            a.eval_status = eval_status;
            reloc_action(a, time_values);
        }
    }
    else {
        std::ranges::sort(artiq_actions, [&] (const auto &a1, const auto &a2) {
            if (a1.reloc_id >= 0 && a1.eval_status != eval_status) {
                a1.eval_status = eval_status;
                reloc_action(a1, time_values);
            }
            if (a2.reloc_id >= 0 && a2.eval_status != eval_status) {
                a2.eval_status = eval_status;
                reloc_action(a2, time_values);
            }
            auto to_tuple = [] (const auto &a) {
                // Move disabled actions to the end,
                // Sort by time (sort by tid for same time value)
                // End action technically happens just before the time point
                // and must be sorted to be before the start action.
                return std::tuple(-int(a.cond), a.time_mu, a.tid, -int(a.is_end), a.aid);
            };
            return to_tuple(a1) < to_tuple(a2);
        });
    }

    ScopeExit cleanup([&] {
        rtio_gen.finish();
    });
    rtio_gen.start();

    int64_t start_mu = 0;

    // Add all the exact time events first
    for (auto start_trigger: start_triggers) {
        auto time_mu = start_trigger.time_mu;
        start_mu = std::min(time_mu, start_mu);
        if (!rtio_gen.time_checker.check_and_add_time(time_mu))
            py_throw_format(PyExc_ValueError,
                            "Too many start triggers at the same time");
        rtio_gen.actions.push_back({start_trigger.target,
                start_trigger.raising_edge, time_mu});
    }
    for (auto start_trigger: start_triggers) {
        auto time_mu = start_trigger.time_mu;
        bb_debug("Adding start trigger: time=%" PRId64 ", raising=%d\n",
                 time_mu, (int)start_trigger.raising_edge);
        if (start_trigger.raising_edge) {
            auto end_mu = time_mu + start_trigger.min_time_mu;
            end_mu = rtio_gen.time_checker.find_time(end_mu, end_mu, end_mu + 1000);
            if (end_mu == INT64_MIN)
                py_throw_format(PyExc_ValueError,
                                "Too many start triggers at the same time");
            rtio_gen.actions.push_back({start_trigger.target, 0, end_mu});
        }
        else {
            auto raise_mu = time_mu - start_trigger.min_time_mu;
            raise_mu = rtio_gen.time_checker.find_time(raise_mu - 1000,
                                                       raise_mu, raise_mu);
            if (raise_mu == INT64_MIN)
                py_throw_format(PyExc_ValueError,
                                "Too many start triggers at the same time");
            rtio_gen.actions.push_back({start_trigger.target, 1, raise_mu});
            start_mu = std::min(raise_mu, start_mu);
        }
    }

    // Add a 3 us buffer to queue outputs before the sequence officially starts
    start_mu -= 3000;

    for (auto &bus: channels.urukul_busses)
        bus.reset(start_mu);
    for (auto &ttlchn: channels.ttlchns)
        ttlchn.reset(start_mu);
    for (auto &ddschn: channels.ddschns)
        ddschn.reset();
    for (const auto &start_value: output->start_values) {
        switch (start_value.type) {
        case DDSFreq:
        case DDSAmp:
        case DDSPhase: {
            auto value = start_value.val_id == -1 ? start_value.value :
                channels.dds_to_mu(start_value.type, start_value.chn_idx,
                                   float_values[start_value.val_id].second);
            auto &ddschn = channels.ddschns[start_value.chn_idx];
            ddschn.had_output = true;
            update_dds_data(ddschn.data1, ddschn.data2, start_value.type, value);
            break;
        }
        case TTLOut:
        case CounterEnable: {
            auto &ttlchn = channels.ttlchns[start_value.chn_idx];
            auto value = ttlchn.ttl_to_mu(start_value.val_id == -1 ?
                                          bool(start_value.value) :
                                          bool_values[start_value.val_id].second);
            ttlchn.cur_val = value;
            ttlchn.new_val = value;
            break;
        }
        }
    }

    for (auto &artiq_action: artiq_actions) {
        // All disabled actions should be at the end.
        if (!artiq_action.cond)
            break;
        auto time_mu = artiq_action.time_mu;
        for (auto &ttlchn: channels.ttlchns)
            ttlchn.flush_output(rtio_gen, time_mu, true, false);
        for (auto &ttlchn: channels.ttlchns)
            ttlchn.flush_output(rtio_gen, time_mu, false, false);
        for (auto &bus: channels.urukul_busses)
            bus.flush_output(rtio_gen, time_mu, false);

        switch (artiq_action.type) {
        case DDSFreq:
        case DDSAmp:
        case DDSPhase: {
            auto &ddschn = channels.ddschns[artiq_action.chn_idx];
            channels.urukul_busses[ddschn.bus_id].add_output(rtio_gen, artiq_action, ddschn);
            break;
        }
        case TTLOut:
        case CounterEnable:
            channels.ttlchns[artiq_action.chn_idx].add_output(rtio_gen, artiq_action);
            break;
        }
    }
    for (auto &ttlchn: channels.ttlchns)
        ttlchn.flush_output(rtio_gen, 0, true, true);
    for (auto &ttlchn: channels.ttlchns)
        ttlchn.flush_output(rtio_gen, 0, false, true);
    for (auto &bus: channels.urukul_busses)
        bus.flush_output(rtio_gen, 0, true);

    std::ranges::stable_sort(rtio_gen.actions, [] (auto &a1, auto &a2) {
        return a1.time_mu < a2.time_mu;
    });

    auto total_time_mu = seq_time_to_mu(cbseq.total_time + max_delay);
    if (use_dma) {
        auto nactions = rtio_gen.actions.size();
        // Note that the size calculated below is at least `nactions * 17 + 1`
        // which is what we need.
        auto alloc_size = (nactions * 17 / 64 + 1) * 64;
        py::bytearray ba = output->rtios;
        ba.resize(alloc_size);
        auto output_ptr = (uint8_t*)ba.data();
        for (size_t i = 0; i < nactions; i++) {
            auto &action = rtio_gen.actions[i];
            auto ptr = &output_ptr[i * 17];
            auto time_mu = action.time_mu - start_mu;
            auto target = action.target;
            auto value = action.value;
            ptr[0] = 17;
            ptr[1] = (target >> 8) & 0xff;
            ptr[2] = (target >> 16) & 0xff;
            ptr[3] = (target >> 24) & 0xff;
            ptr[4] = (time_mu >> 0) & 0xff;
            ptr[5] = (time_mu >> 8) & 0xff;
            ptr[6] = (time_mu >> 16) & 0xff;
            ptr[7] = (time_mu >> 24) & 0xff;
            ptr[8] = (time_mu >> 32) & 0xff;
            ptr[9] = (time_mu >> 40) & 0xff;
            ptr[10] = (time_mu >> 48) & 0xff;
            ptr[11] = (time_mu >> 56) & 0xff;
            ptr[12] = (target >> 0) & 0xff;
            ptr[13] = (value >> 0) & 0xff;
            ptr[14] = (value >> 8) & 0xff;
            ptr[15] = (value >> 16) & 0xff;
            ptr[16] = (value >> 24) & 0xff;
        }
        if (nactions)
            total_time_mu = std::max(rtio_gen.actions.back().time_mu, total_time_mu);
        memset(&output_ptr[nactions * 17], 0, alloc_size - nactions * 17);
    }
    else {
        auto rtio_array = (PyArrayObject*)output->rtios;

        npy_intp rtio_alloc = (npy_intp)PyArray_SIZE(rtio_array);
        PyArray_Dims pydims{&rtio_alloc, 1};
        npy_intp rtio_len = 0;
        uint32_t *rtio_array_data = (uint32_t*)PyArray_DATA(rtio_array);
        auto resize_rtio = [&] (npy_intp sz) {
            rtio_alloc = sz;
            throw_if_not(PyArray_Resize(rtio_array, &pydims, 0, NPY_CORDER));
        };

        auto alloc_space = [&] (int n) -> uint32_t* {
            auto old_len = rtio_len;
            auto new_len = old_len + n;
            if (new_len > rtio_alloc) {
                resize_rtio(new_len * 2 + 2);
                rtio_array_data = (uint32_t*)PyArray_DATA(rtio_array);
            }
            rtio_len = new_len;
            return &rtio_array_data[old_len];
        };

        auto add_wait = [&] (int64_t t_mu) {
            const int64_t max_wait = 0x80000000;
            if (t_mu > max_wait) {
                auto nwait_eles = t_mu / max_wait;
                t_mu = t_mu % max_wait;
                auto wait_cmds = alloc_space(nwait_eles);
                for (int i = 0; i < nwait_eles; i++) {
                    wait_cmds[i] = uint32_t(max_wait);
                }
            }
            if (t_mu > 0) {
                alloc_space(1)[0] = uint32_t(-t_mu);
            }
        };

        int64_t time_mu = start_mu;
        for (auto action: rtio_gen.actions) {
            add_wait(action.time_mu - time_mu);
            time_mu = action.time_mu;
            auto cmd = alloc_space(2);
            cmd[0] = action.target;
            cmd[1] = action.value;
        }
        if (total_time_mu > time_mu) {
            add_wait(total_time_mu - time_mu);
        }
        else {
            total_time_mu = time_mu;
        }
        resize_rtio(rtio_len);
    }
    output->total_time_mu = total_time_mu - start_mu;
}

__attribute__((visibility("internal")))
void ArtiqBackend::Data::runtime_finalize(py::ptr<SeqCompiler> comp, unsigned age)
{
    bb_debug("artiq_runtime_finalize: start\n");
    for (size_t i = 0, nreloc = bool_values.size(); i < nreloc; i++) {
        auto &[rtval, val] = bool_values[i];
        val = !rtval::rtval_cache(rtval).is_zero();
    }
    for (size_t i = 0, nreloc = float_values.size(); i < nreloc; i++) {
        auto &[rtval, val] = float_values[i];
        val = rtval::rtval_cache(rtval).get<double>();
    }
    max_delay = 0;
    auto relocate_delay = [&] (int64_t &delay, rtval_ptr rt_delay) {
        if (!rt_delay)
            return;
        rtval::rt_eval_throw(rt_delay, age);
        delay = convert_device_delay(rtval::rtval_cache(rt_delay).get<double>());
        max_delay = std::max(max_delay, delay);
    };
    for (auto &ttlchn: channels.ttlchns)
        relocate_delay(ttlchn.delay, ttlchn.rt_delay);
    for (auto &ddschn: channels.ddschns)
        relocate_delay(ddschn.delay, ddschn.rt_delay);

    eval_status = !eval_status;
    for (auto [cbseq_id, output]: py::list_iter<Output>(all_outputs))
        generate_bseq(comp, comp->basic_cseqs[cbseq_id], output);

    bb_debug("artiq_runtime_finalize: finish\n");
}

__attribute__((visibility("internal")))
void ArtiqBackend::Data::add_start_trigger(py::ptr<> name, py::ptr<> time,
                                           py::ptr<> min_time, py::ptr<> raising_edge)
{
    auto dev = info().get_device(sys, name);
    if (!dev.isinstance(info().TTLOut))
        py_throw_format(PyExc_ValueError, "Invalid start trigger device: %S", name);
    add_start_trigger_ttl(dev.attr("target_o"_py).as_int(),
                          event_time::round_time_int(time),
                          event_time::round_time_int(min_time),
                          raising_edge.as_bool());
}

BB_PROTECTED
PyTypeObject ArtiqBackend::Type = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "brassboard_seq.artiq_backend.ArtiqBackend",
    .tp_basicsize = sizeof(BackendBase) + sizeof(ArtiqBackend::Data),
    .tp_dealloc = py::tp_cxx_dealloc<true,ArtiqBackend>,
    .tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_HAVE_GC,
    .tp_traverse = traverse<>,
    .tp_clear = clear<>,
    .tp_methods = (
        py::meth_table<
        py::meth_fast<"add_start_trigger",[] (py::ptr<ArtiqBackend> self,
                                              PyObject *const *args, Py_ssize_t nargs) {
            py::check_num_arg("ArtiqBackend.add_start_trigger", nargs, 4, 4);
            self->data()->add_start_trigger(args[0], args[1], args[2], args[3]);
        }>,
        py::meth_fast<"set_device_delay",[] (py::ptr<ArtiqBackend> self,
                                             PyObject *const *args, Py_ssize_t nargs) {
            py::check_num_arg("ArtiqBackend.set_device_delay", nargs, 2, 2);
            py::ptr name = args[0];
            py::ptr delay = args[1];
            if (rtval::is_rtval(delay)) {
                self->data()->device_delay.set(name, delay);
                return;
            }
            auto delay_mu = convert_device_delay(delay.as_float());
            self->data()->device_delay.set(name, to_py(delay_mu));
        }>>),
    .tp_getset = (py::getset_table<
                  py::getset_def<"total_time_mu",[] (py::ptr<ArtiqBackend> self) {
                      auto d = self->data();
                      if (d->all_outputs.size() == 0)
                          return to_py(0);
                      return to_py(d->all_outputs.get<Output>(0)->total_time_mu);
                  }>,
                  py::getset_def<"output",[] (py::ptr<ArtiqBackend> self) {
                      return self->data()->all_outputs.ref();
                  }>>),
    .tp_base = &BackendBase::Type,
    .tp_vectorcall = py::vectorfunc<[] (auto, PyObject *const *args,
                                        ssize_t nargs, py::tuple kwnames) {
        py::check_num_arg("ArtiqBackend.__init__", nargs, 1, 2);
        auto [output_format] =
            py::parse_pos_or_kw_args<"output_format">("ArtiqBackend.__init__",
                                                      args + nargs, 0, kwnames);
        bool use_dma = false;
        if (output_format) {
            auto fmt_str = py::arg_cast<py::str>(output_format, "output_format");
            if (fmt_str.compare_ascii("dma") == 0) {
                use_dma = true;
            }
            else if (fmt_str.compare_ascii("bytecode") != 0) {
                py_throw_format(PyExc_ValueError, "Unknown output type: '%U'",
                                output_format);
            }
        }
        bool support_branch = false;
        auto all_outputs = py::new_list(0);
        if (nargs >= 2) {
            py::ptr rtios;
            if (use_dma) {
                rtios = py::arg_cast<py::bytearray>(args[1], "rtio_array");
            }
            else {
                if (!PyArray_Check(args[1]))
                    py_throw_format(PyExc_TypeError, "Unexpected type '%S' for rtio_array",
                                    Py_TYPE(args[1]));
                auto array = (PyArrayObject*)args[1];
                if (PyArray_NDIM(array) != 1)
                    py_throw_format(PyExc_ValueError, "RTIO output must be a 1D array");
                if (PyArray_TYPE(array) != NPY_INT32)
                    py_throw_format(PyExc_TypeError, "RTIO output must be a int32 array");
                rtios = array;
            }
            all_outputs.append(Output::alloc(0, rtios.ref(), true, {}));
        }
        else {
            support_branch = true;
        }
        return alloc(args[0], std::move(all_outputs), use_dma, support_branch);
    }>
};

__attribute__((visibility("hidden")))
void init()
{
    _import_array();
    throw_if(PyType_Ready(&ArtiqBackend::Output::Type) < 0);
    throw_if(PyType_Ready(&ArtiqBackend::Type) < 0);
}

}
