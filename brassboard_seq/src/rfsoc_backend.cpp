//

#include "rfsoc_backend.h"
#include "utils.h"

#include <assert.h>

namespace rfsoc_backend {

inline int ChannelInfo::add_tone_channel(int chn)
{
    int chn_idx = (int)channels.size();
    channels.push_back({ chn });
    return chn_idx;
}

inline void ChannelInfo::add_seq_channel(int seq_chn, int chn_idx, ToneParam param)
{
    assert(chn_map.count(seq_chn) == 0);
    chn_map.insert({seq_chn, {chn_idx, param}});
}

struct CompileVTable {
    int (*is_rtval)(PyObject*);
    int (*is_ramp)(PyObject*);
};

static inline bool parse_action_kws(PyObject *kws, int aid)
{
    if (kws == Py_None)
        return false;
    bool sync = false;
    PyObject *key, *value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(kws, &pos, &key, &value)) {
        if (PyUnicode_CompareWithASCIIString(key, "sync") == 0) {
            sync = get_value_bool(value, action_key(aid));
            continue;
        }
        bb_err_format(PyExc_ValueError, action_key(aid),
                      "Invalid output keyword argument %S", kws);
        throw 0;
    }
    return sync;
}

template<typename Action, typename EventTime, typename RFSOCBackend>
static __attribute__((always_inline)) inline
void collect_actions(RFSOCBackend *rb, const CompileVTable vtable, Action*, EventTime*)
{
    auto seq = rb->__pyx_base.seq;
    auto all_actions = seq->all_actions;

    ValueIndexer<int> bool_values;
    ValueIndexer<double> float_values;
    std::vector<Relocation> &relocations = rb->relocations;
    auto event_times = seq->__pyx_base.__pyx_base.seqinfo->time_mgr->event_times;

    for (auto [seq_chn, value]: rb->channels.chn_map) {
        auto [chn_idx, param] = value;
        auto is_ff = param == ToneFF;
        auto &channel = rb->channels.channels[chn_idx];
        auto &rfsoc_actions = channel.actions[(int)param];
        auto actions = PyList_GET_ITEM(all_actions, seq_chn);
        auto nactions = PyList_GET_SIZE(actions);
        for (int idx = 0; idx < nactions; idx++) {
            auto action = (Action*)PyList_GET_ITEM(actions, idx);
            auto sync = parse_action_kws(action->kws, action->aid);
            auto value = action->value;
            auto is_ramp = vtable.is_ramp(value);
            if (is_ff && is_ramp) {
                bb_err_format(PyExc_ValueError, action_key(action->aid),
                              "Feed forward control cannot be ramped");
                throw 0;
            }
            auto cond = action->cond;
            if (cond == Py_False)
                continue;
            bool cond_need_reloc = vtable.is_rtval(cond);
            assert(cond_need_reloc || cond == Py_True);
            int cond_idx = cond_need_reloc ? bool_values.get_id(cond) : -1;
            auto add_action = [&] (auto value, int tid, bool sync, bool is_ramp,
                                   bool is_end) {
                bool needs_reloc = cond_need_reloc;
                Relocation reloc{cond_idx, -1, -1};

                RFSOCAction rfsoc_action;
                rfsoc_action.cond = !cond_need_reloc;
                rfsoc_action.eval_status = false;
                rfsoc_action.isramp = is_ramp;
                rfsoc_action.sync = sync;
                rfsoc_action.aid = action->aid;
                rfsoc_action.tid = tid;
                rfsoc_action.is_end = is_end;

                auto event_time = (EventTime*)PyList_GET_ITEM(event_times, tid);
                if (event_time->data.has_static) {
                    rfsoc_action.seq_time = event_time->data._get_static();
                }
                else {
                    needs_reloc = true;
                    reloc.time_idx = tid;
                }
                if (is_ramp) {
                    rfsoc_action.ramp = value;
                    auto len = action->length;
                    if (vtable.is_rtval(len)) {
                        needs_reloc = true;
                        reloc.val_idx = float_values.get_id(len);
                    }
                    else {
                        rfsoc_action.float_value =
                            get_value_f64(len, action_key(action->aid));
                    }
                }
                else if (vtable.is_rtval(value)) {
                    needs_reloc = true;
                    if (is_ff) {
                        reloc.val_idx = bool_values.get_id(value);
                    }
                    else {
                        reloc.val_idx = float_values.get_id(value);
                    }
                }
                else if (is_ff) {
                    rfsoc_action.bool_value = get_value_bool(value,
                                                             action_key(action->aid));
                }
                else {
                    rfsoc_action.float_value = get_value_f64(value,
                                                             action_key(action->aid));
                }
                if (needs_reloc) {
                    rfsoc_action.reloc_id = (int)relocations.size();
                    relocations.push_back(reloc);
                }
                else {
                    rfsoc_action.reloc_id = -1;
                }
                rfsoc_actions.push_back(rfsoc_action);
            };
            add_action(action->value, action->tid, sync, is_ramp, false);
            if (action->data.is_pulse || is_ramp) {
                add_action(action->end_val, action->end_tid, false, false, true);
            }
        }
    }

    rb->bool_values = std::move(bool_values.values);
    rb->float_values = std::move(float_values.values);
}

}
