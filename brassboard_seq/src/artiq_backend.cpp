//

#include "artiq_backend.h"
#include "utils.h"

#include <algorithm>

#include <assert.h>

namespace artiq_backend {
static ArtiqConsts artiq_consts;

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

inline void ChannelsInfo::add_ttl_channel(int seqchn, uint32_t target, bool iscounter)
{
    assert(ttl_chn_map.count(seqchn) == 0);
    auto ttl_id = (int)ttlchns.size();
    ttlchns.push_back({target, iscounter});
    ttl_chn_map[seqchn] = ttl_id;
}

inline int ChannelsInfo::get_dds_channel_id(uint32_t bus_id, double ftw_per_hz,
                                            uint8_t chip_select)
{
    std::pair<int,int> key{bus_id, chip_select};
    auto it = dds_chn_map.find(key);
    if (it != dds_chn_map.end())
        return it->second;
    auto dds_id = (int)ddschns.size();
    ddschns.push_back({ftw_per_hz, bus_id, chip_select});
    dds_chn_map[key] = dds_id;
    return dds_id;
}

inline void ChannelsInfo::add_dds_param_channel(int seqchn, uint32_t bus_id,
                                                double ftw_per_hz, uint8_t chip_select,
                                                ChannelType param)
{
    assert(dds_param_chn_map.count(seqchn) == 0);
    dds_param_chn_map[seqchn] = {get_dds_channel_id(bus_id, ftw_per_hz,
                                                    chip_select), param};
}

struct CompileVTable {
    int (*is_rtval)(PyObject*);
    int (*is_ramp)(PyObject*);
};

template<typename Action, typename EventTime, typename ArtiqBackend>
static __attribute__((always_inline)) inline
void collect_actions(ArtiqBackend *ab, const CompileVTable vtable, Action*, EventTime*)
{
    auto seq = ab->__pyx_base.seq;
    auto all_actions = seq->all_actions;
    std::vector<ArtiqAction> &artiq_actions = ab->all_actions;

    ValueIndexer<int> bool_values;
    ValueIndexer<double> float_values;
    std::vector<Relocation> &relocations = ab->relocations;

    auto event_times = seq->__pyx_base.__pyx_base.seqinfo->time_mgr->event_times;

    auto add_single_action = [&] (Action *action, ChannelType type, int chn_idx,
                                  int tid, PyObject *value, int cond_reloc,
                                  bool is_end) {
        ArtiqAction artiq_action;
        int aid = action->aid;
        artiq_action.type = type;
        artiq_action.cond = true;
        artiq_action.exact_time = action->data.exact_time;
        artiq_action.eval_status = false;
        artiq_action.chn_idx = chn_idx;
        artiq_action.tid = tid;
        artiq_action.is_end = is_end;
        artiq_action.aid = aid;

        bool needs_reloc = cond_reloc != -1;
        Relocation reloc{cond_reloc, -1, -1};

        auto event_time = (EventTime*)PyList_GET_ITEM(event_times, tid);
        if (event_time->data.has_static) {
            artiq_action.time_mu = seq_time_to_mu(event_time->data._get_static());
        }
        else {
            needs_reloc = true;
            reloc.time_idx = tid;
        }
        if (vtable.is_rtval(value)) {
            needs_reloc = true;
            if (type == DDSFreq || type == DDSAmp || type == DDSPhase) {
                reloc.val_idx = float_values.get_id(value);
            }
            else if (type == CounterEnable) {
                // We aren't really relying on this in the backend
                // but requiring this makes it easier to infer the number of
                // results generated from a sequence.
                bb_err_format(PyExc_ValueError, action_key(aid),
                              "Counter value must be static.");
                throw 0;
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
    auto add_action = [&] (Action *action, ChannelType type, int chn_idx) {
        auto cond = action->cond;
        int cond_reloc = -1;
        if (vtable.is_rtval(cond)) {
            cond_reloc = bool_values.get_id(cond);
            assume(cond_reloc >= 0);
        }
        else {
            assert(cond == Py_True);
        }
        add_single_action(action, type, chn_idx, action->tid, action->value,
                          cond_reloc, false);
        if (action->data.is_pulse) {
            add_single_action(action, type, chn_idx, action->end_tid, action->end_val,
                              cond_reloc, true);
        }
    };

    for (auto [chn, ttl_idx]: ab->channels.ttl_chn_map) {
        auto ttl_chn_info = ab->channels.ttlchns[ttl_idx];
        auto type = ttl_chn_info.iscounter ? CounterEnable : TTLOut;
        auto actions = PyList_GET_ITEM(all_actions, chn);
        auto nactions = PyList_GET_SIZE(actions);
        for (int idx = 0; idx < nactions; idx++) {
            auto action = (Action*)PyList_GET_ITEM(actions, idx);
            if (action->kws != Py_None) {
                bb_err_format(PyExc_ValueError, action_key(action->aid),
                              "Invalid output keyword argument %S", action->kws);
                throw 0;
            }
            auto value = action->value;
            if (vtable.is_ramp(value)) {
                bb_err_format(PyExc_ValueError, action_key(action->aid),
                              "TTL Channel cannot be ramped");
                throw 0;
            }
            if (action->cond == Py_False)
                continue;
            add_action(action, type, ttl_idx);
        }
    }

    for (auto [chn, value]: ab->channels.dds_param_chn_map) {
        auto [dds_idx, type] = value;
        auto actions = PyList_GET_ITEM(all_actions, chn);
        auto nactions = PyList_GET_SIZE(actions);
        for (int idx = 0; idx < nactions; idx++) {
            auto action = (Action*)PyList_GET_ITEM(actions, idx);
            if (action->kws != Py_None) {
                bb_err_format(PyExc_ValueError, action_key(action->aid),
                              "Invalid output keyword argument %S", action->kws);
                throw 0;
            }
            auto value = action->value;
            if (vtable.is_ramp(value)) {
                bb_err_format(PyExc_ValueError, action_key(action->aid),
                              "DDS Channel cannot be ramped");
                throw 0;
            }
            if (action->cond == Py_False)
                continue;
            add_action(action, type, dds_idx);
        }
    }

    ab->bool_values = std::move(bool_values.values);
    ab->float_values = std::move(float_values.values);
}

}
