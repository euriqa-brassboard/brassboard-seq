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
#include "utils.h"

#include <algorithm>

#include <assert.h>

#include <numpy/arrayobject.h>

namespace brassboard_seq::artiq_backend {

using rtval::RuntimeValue;
using event_time::EventTime;

struct ArtiqConsts {
    int COUNTER_ENABLE;
    int COUNTER_DISABLE;
    int _AD9910_REG_PROFILE0;
    int URUKUL_CONFIG;
    int URUKUL_CONFIG_END;
    int URUKUL_SPIT_DDS_WR;
    int URUKUL_DEFAULT_PROFILE;
    int SPI_CONFIG_ADDR;
    int SPI_DATA_ADDR;
};

static ArtiqConsts artiq_consts;

static PyObject *rampfunctionbase_type;

inline int ChannelsInfo::add_bus_channel(int bus_channel, uint32_t io_update_target,
                                         uint8_t ref_period_mu)
{
    auto bus_id = (int)urukul_busses.size();
    urukul_busses.push_back({
            uint32_t(bus_channel),
            uint32_t((bus_channel << 8) | artiq_consts.SPI_CONFIG_ADDR),
            uint32_t((bus_channel << 8) | artiq_consts.SPI_DATA_ADDR),
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

inline void ChannelsInfo::add_ttl_channel(int seqchn, uint32_t target, bool iscounter,
                                          int64_t delay, PyObject *rt_delay)
{
    assert(ttl_chn_map.count(seqchn) == 0);
    auto ttl_id = (int)ttlchns.size();
    ttlchns.push_back({target, iscounter, delay, rt_delay});
    ttl_chn_map[seqchn] = ttl_id;
}

inline int ChannelsInfo::get_dds_channel_id(uint32_t bus_id, double ftw_per_hz,
                                            uint8_t chip_select, int64_t delay,
                                            PyObject *rt_delay)
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

inline void ChannelsInfo::add_dds_param_channel(int seqchn, uint32_t bus_id,
                                                double ftw_per_hz, uint8_t chip_select,
                                                ChannelType param, int64_t delay,
                                                PyObject *rt_delay)
{
    assert(dds_param_chn_map.count(seqchn) == 0);
    dds_param_chn_map[seqchn] = {get_dds_channel_id(bus_id, ftw_per_hz, chip_select,
                                                    delay, rt_delay), param};
}

static __attribute__((always_inline)) inline
void collect_actions(auto *ab, backend::CompiledSeq &cseq)
{
    auto seq = pyx_fld(ab, seq);
    std::vector<ArtiqAction> &artiq_actions = ab->all_actions;

    ValueIndexer<int> bool_values;
    ValueIndexer<double> float_values;
    std::vector<Relocation> &relocations = ab->relocations;

    auto event_times = pyx_fld(seq, seqinfo)->time_mgr->event_times;

    auto add_single_action = [&] (auto *action, ChannelType type, int chn_idx,
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

        auto event_time = py::list(event_times).get<EventTime>(tid);
        if (event_time->data.is_static()) {
            PyObject *rt_delay;
            int64_t delay;
            if (type == DDSFreq || type == DDSAmp || type == DDSPhase) {
                auto &ddschn = ab->channels.ddschns[chn_idx];
                delay = ddschn.delay;
                rt_delay = ddschn.rt_delay;
            }
            else {
                auto &ttlchn = ab->channels.ttlchns[chn_idx];
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
                reloc.val_idx = float_values.get_id(value);
            }
            else if (type == CounterEnable) {
                // We aren't really relying on this in the backend
                // but requiring this makes it easier to infer the number of
                // results generated from a sequence.
                bb_throw_format(PyExc_ValueError, action_key(aid),
                                "Counter value must be static.");
            }
            else {
                reloc.val_idx = bool_values.get_id(value);
            }
        }
        else if (type == DDSFreq || type == DDSAmp || type == DDSPhase) {
            double v = get_value_f64(value, action_key(aid));
            if (type == DDSFreq) {
                auto &ddschn = ab->channels.ddschns[chn_idx];
                artiq_action.value = dds_freq_to_mu(v, ddschn.ftw_per_hz);
            }
            else if (type == DDSAmp) {
                artiq_action.value = dds_amp_to_mu(v);
            }
            else {
                assert(type == DDSPhase);
                artiq_action.value = dds_phase_to_mu(v);
            }
        }
        else {
            artiq_action.value = get_value_bool(value, action_key(aid));
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
    auto add_action = [&] (auto *action, ChannelType type, int chn_idx) {
        auto cond = action->cond.get();
        int cond_reloc = -1;
        if (rtval::is_rtval(cond)) {
            cond_reloc = bool_values.get_id(cond);
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

    for (auto [chn, ttl_idx]: ab->channels.ttl_chn_map) {
        auto ttl_chn_info = ab->channels.ttlchns[ttl_idx];
        auto type = ttl_chn_info.iscounter ? CounterEnable : TTLOut;
        for (auto action: cseq.all_actions[chn]) {
            if (action->kws)
                bb_throw_format(PyExc_ValueError, action_key(action->aid),
                                "Invalid output keyword argument %S",
                                action->kws.get());
            auto value = action->value.get();
            if (py_issubtype_nontrivial(Py_TYPE(value), rampfunctionbase_type))
                bb_throw_format(PyExc_ValueError, action_key(action->aid),
                                "TTL Channel cannot be ramped");
            if (action->cond == Py_False)
                continue;
            add_action(action, type, ttl_idx);
        }
    }

    for (auto [chn, value]: ab->channels.dds_param_chn_map) {
        auto [dds_idx, type] = value;
        for (auto action: cseq.all_actions[chn]) {
            if (action->kws)
                bb_throw_format(PyExc_ValueError, action_key(action->aid),
                                "Invalid output keyword argument %S",
                                action->kws.get());
            auto value = action->value.get();
            if (py_issubtype_nontrivial(Py_TYPE(value), rampfunctionbase_type))
                bb_throw_format(PyExc_ValueError, action_key(action->aid),
                                "DDS Channel cannot be ramped");
            if (action->cond == Py_False)
                continue;
            add_action(action, type, dds_idx);
        }
    }

    ab->bool_values = std::move(bool_values.values);
    ab->float_values = std::move(float_values.values);
}

static __attribute__((always_inline)) inline
void generate_rtios(auto *ab, backend::CompiledSeq &cseq, unsigned age)
{
    bb_debug("generate_rtios: start\n");
    auto seq = pyx_fld(ab, seq);
    for (size_t i = 0, nreloc = ab->bool_values.size(); i < nreloc; i++) {
        auto &[rtval, val] = ab->bool_values[i];
        val = !rtval::rtval_cache(rtval).is_zero();
    }
    for (size_t i = 0, nreloc = ab->float_values.size(); i < nreloc; i++) {
        auto &[rtval, val] = ab->float_values[i];
        val = rtval::rtval_cache(rtval).template get<double>();
    }
    int64_t max_delay = 0;
    auto relocate_delay = [&] (int64_t &delay, auto rt_delay) {
        if (!rt_delay)
            return;
        rtval::rt_eval_throw(rt_delay, age);
        auto fdelay = rtval::rtval_cache(rt_delay).template get<double>();
        if (fdelay < 0) {
            py_throw_format(PyExc_ValueError,
                            "Device time offset %S cannot be negative.",
                            py::new_float(fdelay));
        }
        else if (fdelay > 0.1) {
            py_throw_format(PyExc_ValueError,
                            "Device time offset %S cannot be more than 100ms.",
                            py::new_float(fdelay));
        }
        delay = int64_t(fdelay * event_time::time_scale + 0.5);
    };
    for (auto &ttlchn: ab->channels.ttlchns) {
        relocate_delay(ttlchn.delay, (RuntimeValue*)ttlchn.rt_delay);
        max_delay = std::max(max_delay, ttlchn.delay);
    }
    for (auto &ddschn: ab->channels.ddschns) {
        relocate_delay(ddschn.delay, (RuntimeValue*)ddschn.rt_delay);
        max_delay = std::max(max_delay, ddschn.delay);
    }
    auto &time_values = pyx_fld(seq, seqinfo)->time_mgr->time_values;

    auto reloc_action = [ab, &time_values] (const ArtiqAction &action) {
        auto reloc = ab->relocations[action.reloc_id];
        if (reloc.cond_idx != -1)
            action.cond = ab->bool_values[reloc.cond_idx].second;
        // No need to do anything else if we hit a disabled action.
        if (!action.cond)
            return;
        auto type = action.type;
        auto chn_idx = action.chn_idx;
        if (reloc.time_idx != -1) {
            int64_t delay;
            if (type == DDSFreq || type == DDSAmp || type == DDSPhase) {
                delay = ab->channels.ddschns[chn_idx].delay;
            }
            else {
                delay = ab->channels.ttlchns[chn_idx].delay;
            }
            action.time_mu = seq_time_to_mu(time_values[reloc.time_idx] + delay);
        }
        if (reloc.val_idx != -1) {
            if (type == DDSFreq || type == DDSAmp || type == DDSPhase) {
                double v = ab->float_values[reloc.val_idx].second;
                if (type == DDSFreq) {
                    auto &ddschn = ab->channels.ddschns[chn_idx];
                    action.value = dds_freq_to_mu(v, ddschn.ftw_per_hz);
                }
                else if (type == DDSAmp) {
                    action.value = dds_amp_to_mu(v);
                }
                else {
                    assert(type == DDSPhase);
                    action.value = dds_phase_to_mu(v);
                }
            }
            else {
                action.value = ab->bool_values[reloc.val_idx].second;
            }
        }
    };

    bool eval_status = !ab->eval_status;
    ab->eval_status = eval_status;

    if (ab->all_actions.size() == 1) {
        auto &a = ab->all_actions[0];
        if (a.reloc_id >= 0) {
            a.eval_status = eval_status;
            reloc_action(a);
        }
    }
    else {
        std::ranges::sort(ab->all_actions, [&] (const auto &a1, const auto &a2) {
            if (a1.reloc_id >= 0 && a1.eval_status != eval_status) {
                a1.eval_status = eval_status;
                reloc_action(a1);
            }
            if (a2.reloc_id >= 0 && a2.eval_status != eval_status) {
                a2.eval_status = eval_status;
                reloc_action(a2);
            }
            // Move disabled actions to the end
            if (a1.cond != a2.cond)
                return int(a1.cond) > int(a2.cond);
            // Sort by time
            if (a1.time_mu != a2.time_mu)
                return a1.time_mu < a2.time_mu;
            // Sometimes time points with different tid may actually happen
            // at the same time (especially on the same artiq mu time point)
            // in these cases we need to make sure we output them in order
            // in order to generate the right output at the end.
            if (a1.tid != a2.tid)
                return a1.tid < a2.tid;
            // End action technically happens just before the time point
            // and must be sorted to be before the start action.
            if (a1.is_end != a2.is_end)
                return int(a1.is_end) > int(a2.is_end);
            return a1.aid < a2.aid;
        });
    }

    auto &rtio_actions = ab->rtio_actions;
    ScopeExit cleanup([&] {
        ab->time_checker.clear();
        rtio_actions.clear();
    });
    assert(ab->time_checker.empty());
    assert(rtio_actions.empty());

    auto add_action = [&] (uint32_t target, uint32_t value, int aid,
                           int64_t request_time_mu, int64_t lb_mu,
                           bool exact_time) -> int64_t {
        if (exact_time) {
            if (request_time_mu < lb_mu) {
                bb_throw_format(PyExc_ValueError, action_key(aid),
                                "Exact time output cannot satisfy lower time bound");
            }
            else if (!ab->time_checker.check_and_add_time(request_time_mu)) {
                bb_throw_format(PyExc_ValueError, action_key(aid),
                                "Too many outputs at the same time");
            }
            rtio_actions.push_back({target, value, request_time_mu});
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
        auto time_mu = ab->time_checker.find_time(lb_mu, request_time_mu, ub_mu);
        if (time_mu == INT64_MIN)
            bb_throw_format(PyExc_ValueError, action_key(aid),
                            "Too many outputs at the same time");
        rtio_actions.push_back({target, value, time_mu});
        return time_mu;
    };

    int64_t start_mu = 0;

    // Add all the exact time events first
    for (auto start_trigger: ab->start_triggers) {
        auto time_mu = start_trigger.time_mu;
        start_mu = std::min(time_mu, start_mu);
        if (!ab->time_checker.check_and_add_time(time_mu))
            py_throw_format(PyExc_ValueError,
                            "Too many start triggers at the same time");
        rtio_actions.push_back({start_trigger.target,
                start_trigger.raising_edge, time_mu});
    }
    for (auto start_trigger: ab->start_triggers) {
        auto time_mu = start_trigger.time_mu;
        bb_debug("Adding start trigger: time=%" PRId64 ", raising=%d\n",
                 time_mu, (int)start_trigger.raising_edge);
        if (start_trigger.raising_edge) {
            auto end_mu = time_mu + start_trigger.min_time_mu;
            end_mu = ab->time_checker.find_time(end_mu, end_mu, end_mu + 1000);
            if (end_mu == INT64_MIN)
                py_throw_format(PyExc_ValueError,
                                "Too many start triggers at the same time");
            rtio_actions.push_back({start_trigger.target, 0, end_mu});
        }
        else {
            auto raise_mu = time_mu - start_trigger.min_time_mu;
            raise_mu = ab->time_checker.find_time(raise_mu - 1000, raise_mu, raise_mu);
            if (raise_mu == INT64_MIN)
                py_throw_format(PyExc_ValueError,
                                "Too many start triggers at the same time");
            rtio_actions.push_back({start_trigger.target, 1, raise_mu});
            start_mu = std::min(raise_mu, start_mu);
        }
    }

    // Add a 3 us buffer to queue outputs before the sequence officially starts
    start_mu -= 3000;

    for (auto &bus: ab->channels.urukul_busses)
        bus.reset(start_mu);
    for (auto &ttlchn: ab->channels.ttlchns)
        ttlchn.reset(start_mu);
    for (auto &ddschn: ab->channels.ddschns)
        ddschn.reset();

    for (auto &artiq_action: ab->all_actions) {
        // All disabled actions should be at the end.
        if (!artiq_action.cond)
            break;
        auto time_mu = artiq_action.time_mu;
        for (auto &ttlchn: ab->channels.ttlchns)
            ttlchn.flush_output(add_action, time_mu, true, false);
        for (auto &ttlchn: ab->channels.ttlchns)
            ttlchn.flush_output(add_action, time_mu, false, false);
        for (auto &bus: ab->channels.urukul_busses)
            bus.flush_output(add_action, time_mu, false);

        switch (artiq_action.type) {
        case DDSFreq:
        case DDSAmp:
        case DDSPhase: {
            auto &ddschn = ab->channels.ddschns[artiq_action.chn_idx];
            ab->channels.urukul_busses[ddschn.bus_id].add_output(add_action,
                                                                 artiq_action, ddschn);
            break;
        }
        case TTLOut:
        case CounterEnable:
            ab->channels.ttlchns[artiq_action.chn_idx].add_output(add_action,
                                                                  artiq_action);
            break;
        }
    }
    for (auto &ttlchn: ab->channels.ttlchns)
        ttlchn.flush_output(add_action, 0, true, true);
    for (auto &ttlchn: ab->channels.ttlchns)
        ttlchn.flush_output(add_action, 0, false, true);
    for (auto &bus: ab->channels.urukul_busses)
        bus.flush_output(add_action, 0, true);


    std::ranges::stable_sort(rtio_actions, [] (auto &a1, auto &a2) {
        return a1.time_mu < a2.time_mu;
    });

    auto total_time_mu = seq_time_to_mu(cseq.total_time + max_delay);
    if (ab->use_dma) {
        auto rtio_array = ab->rtio_array;
        auto nactions = rtio_actions.size();
        // Note that the size calculated below is at least `nactions * 17 + 1`
        // which is what we need.
        auto alloc_size = (nactions * 17 / 64 + 1) * 64;
        PyByteArray_Resize(rtio_array, alloc_size);
        auto output_ptr = (uint8_t*)PyByteArray_AS_STRING(rtio_array);
        for (size_t i = 0; i < nactions; i++) {
            auto &action = rtio_actions[i];
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
            total_time_mu = std::max(rtio_actions.back().time_mu, total_time_mu);
        memset(&output_ptr[nactions * 17], 0, alloc_size - nactions * 17);
    }
    else {
        auto rtio_array = (PyArrayObject*)ab->rtio_array;

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
                auto wait_cmd = alloc_space(1);
                wait_cmd[0] = uint32_t(-t_mu);
            }
        };

        int64_t time_mu = start_mu;
        for (auto action: rtio_actions) {
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
    ab->total_time_mu = total_time_mu - start_mu;

    bb_debug("generate_rtios: finish\n");
    return;
}

void UrukulBus::add_dds_action(auto &add_action, DDSAction &action)
{
    auto div = artiq_consts.URUKUL_SPIT_DDS_WR;
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
        auto t1 = add_action(addr_target, addr, action.aid, lb_mu1, lb_mu1, false);
        lb_mu2 = std::max(lb_mu2, t1 + ref_period_mu);
        auto t2 = add_action(data_target, data, action.aid, lb_mu2, lb_mu2, false);
        return t2 + data_time_mu;
    };
    auto profile_reg = (artiq_consts._AD9910_REG_PROFILE0 +
                        artiq_consts.URUKUL_DEFAULT_PROFILE);
    // We can't start the write safely before the previous io_update finishes
    // since it may abort the write. However, we can configure the SPI controller
    // before the io_update finishes since no signal should be sent to the DDS.
    auto t1 = config_and_write(artiq_consts.URUKUL_CONFIG, 8, profile_reg << 24,
                               last_bus_mu, std::max(last_bus_mu, last_io_update_mu));
    auto t2 = config_and_write(artiq_consts.URUKUL_CONFIG, 32, action.data1, t1, t1);
    auto t3 = config_and_write(artiq_consts.URUKUL_CONFIG_END, 32, action.data2,
                               t2, t2);
    ddschn->data1 = action.data1;
    ddschn->data2 = action.data2;
    ddschn->had_output = true;
    last_bus_mu = t3;
}

void UrukulBus::add_io_update(auto &add_action, int64_t time_mu,
                              int aid, bool exact_time)
{
    // Round to the nearest 8 cycles.
    time_mu = (time_mu + coarse_time_mu / 2) & ~int64_t(coarse_time_mu - 1);
    bb_debug("add_io_update: aid=%d, bus@%" PRId64 ", io_upd@%" PRId64 ", "
             "time=%" PRId64 ", exact_time=%d, chn=%d\n",
             aid, last_bus_mu, last_io_update_mu, time_mu, exact_time, channel);
    auto t1 = add_action(io_update_target, 1, aid, time_mu,
                         std::max(last_bus_mu, last_io_update_mu), exact_time);
    auto t2 = add_action(io_update_target, 0, aid, t1 + coarse_time_mu,
                         t1 + coarse_time_mu, false);
    last_io_update_mu = t2;
}

inline void UrukulBus::flush_output(auto &add_action, int64_t time_mu, bool force)
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
        add_dds_action(add_action, action);
        if (action.exact_time) {
            add_io_update(add_action, action.time_mu, action.aid, true);
        }
        else {
            need_io_update = true;
            aid = action.aid;
            update_time_mu = action.time_mu;
        }
        dds_actions.erase(dds_actions.begin());
    }
    if (need_io_update) {
        add_io_update(add_action, update_time_mu, aid, false);
    }
}

inline void UrukulBus::add_output(auto &add_action, const ArtiqAction &action,
                                  DDSChannel &ddschn)
{
    auto update_dds_data = [&] (uint32_t &data1, uint32_t &data2) {
        if (action.type == DDSFreq) {
            data2 = action.value;
        }
        else if (action.type == DDSAmp) {
            data1 = (data1 & 0xffff) | (action.value << 16);
        }
        else {
            assert(action.type == DDSPhase);
            data1 = (data1 & 0xffff0000) | action.value;
        }
    };
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
                    add_dds_action(add_action, dds_action2);
                    if (action.exact_time) {
                        add_io_update(add_action, dds_action2.time_mu,
                                      dds_action2.aid, true);
                    }
                }
                dds_actions.erase(dds_actions.begin(), last);
                break;
            }
            update_dds_data(dds_action.data1, dds_action.data2);
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
    update_dds_data(data1, data2);
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

inline void TTLChannel::flush_output(auto &add_action, int64_t cur_time_mu,
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
    last_time_mu = add_action(target, new_val, aid, time_mu,
                              last_time_mu, exact_time) + coarse_time_mu;
}

inline void TTLChannel::add_output(auto &add_action, const ArtiqAction &action)
{
    uint8_t val;
    if (!iscounter) {
        val = action.value;
    }
    else if (action.value) {
        val = artiq_consts.COUNTER_ENABLE;
    }
    else {
        val = artiq_consts.COUNTER_DISABLE;
    }

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
        last_time_mu = add_action(target, val, action.aid, action.time_mu,
                                  last_time_mu, action.exact_time) + coarse_time_mu;
        return;
    }

    // If we have two exact outputs we need to flush the previous one
    if (cur_val != new_val && exact_time && action.exact_time &&
        action.time_mu != time_mu) {
        bb_debug("add_ttl: flush pending exact time output\n");
        last_time_mu = add_action(target, new_val, aid, time_mu,
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

inline void TimeChecker::clear()
{
    counts.clear();
    max_key = 0;
}

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

static rtval::TagVal evalonce_callback(auto *self)
{
    if (self->value == Py_None)
        py_throw_format(PyExc_RuntimeError, "Value evaluated too early");
    return rtval::TagVal::from_py(self->value);
}

}
