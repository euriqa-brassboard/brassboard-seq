//

#include "rfsoc_backend.h"
#include "utils.h"

#include <algorithm>

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

static inline
const char *param_name(int param)
{
    switch (param) {
    case ToneFreq:
        return "freq";
    case ToneAmp:
        return "amp";
    case TonePhase:
        return "phase";
    case ToneFF:
    default:
        return "ff";
    }
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

struct RuntimeVTable {
    PyObject *(*rt_eval)(PyObject*, unsigned age);
    int (*rampbuffer_eval_segments)(PyObject *buff, PyObject *func, PyObject *length,
                                    PyObject *oldval, double **input, double **output);
    double *(*rampbuffer_alloc_input)(PyObject *buff, int size);
    double *(*rampbuffer_eval)(PyObject *buff, PyObject *func,
                               PyObject *length, PyObject *oldval);
    int (*ramp_get_cubic_spline)(PyObject*, cubic_spline_t *sp);
};

template<typename RFSOCBackend>
static __attribute__((noreturn))
void reraise_reloc_error(RFSOCBackend *rb, size_t reloc_idx, bool isbool)
{
    // This is inefficient when we actually hit an error but saves memory
    // when we didn't have an error
    for (auto &channel: rb->channels.channels) {
        for (int param = 0; param < _NumToneParam; param++) {
            auto action_isbool = param == ToneFF;
            if (action_isbool != isbool)
                continue;
            for (auto &rfsoc_action: channel.actions[param]) {
                if (rfsoc_action.reloc_id == -1)
                    continue;
                auto reloc = rb->relocations[rfsoc_action.reloc_id];
                // We only need to check for the value index since the time relocation
                // does not use our relocation table as the input and the conditional
                // values are checked by the sequence common code
                // (and their value is cached so there shouldn't be an error
                // when we try to evaluate it.)
                bb_reraise_and_throw_if(reloc.val_idx == reloc_idx,
                                        action_key(rfsoc_action.aid));
            }
        }
    }
    throw 0;
}

static constexpr double min_spline_time = 150e-9;

struct SplineBuffer {
    double t[7];
    double v[7];

    bool is_accurate_enough(double threshold) const
    {
        if (t[3] - t[0] <= min_spline_time)
            return true;
        auto sp = spline_from_values(v[0], v[2], v[4], v[6]);
        if (abs(spline_eval(sp, 1.0 / 6) - v[1]) > threshold)
            return false;
        if (abs(spline_eval(sp, 1.0 / 2) - v[3]) > threshold)
            return false;
        if (abs(spline_eval(sp, 5.0 / 6) - v[5]) > threshold)
            return false;
        return true;
    }
};

template<typename EvalCB, typename AddSample>
static __attribute__((flatten))
void _generate_splines(EvalCB &eval_cb, AddSample &add_sample,
                       SplineBuffer &buff, double threshold)
{
    bb_debug("generate_splines: {%f, %f, %f, %f, %f, %f, %f} -> "
             "{%f, %f, %f, %f, %f, %f, %f}\n",
             buff.t[0], buff.t[1], buff.t[2], buff.t[3],
             buff.t[4], buff.t[5], buff.t[6],
             buff.v[0], buff.v[1], buff.v[2], buff.v[3],
             buff.v[4], buff.v[5], buff.v[6]);
    if (buff.is_accurate_enough(threshold)) {
        bb_debug("accurate enough: t0=%f, t1=%f, t2=%f\n",
                 buff.t[0], buff.t[3], buff.t[6]);
        add_sample(buff.t[3], buff.v[0], buff.v[1], buff.v[2], buff.v[3]);
        add_sample(buff.t[6], buff.v[3], buff.v[4], buff.v[5], buff.v[6]);
        return;
    }
    {
        SplineBuffer buff0;
        {
            double ts[6];
            for (int i = 0; i < 6; i++)
                ts[i] = (buff.t[i] + buff.t[i + 1]) / 2;
            bb_debug("evaluate on {%f, %f, %f, %f, %f, %f}\n",
                     ts[0], ts[1], ts[2], ts[3], ts[4], ts[5]);
            auto vs = eval_cb(6, ts);
            buff0 = {
                { buff.t[0], ts[0], buff.t[1], ts[1], buff.t[2], ts[2], buff.t[3] },
                { buff.v[0], vs[0], buff.v[1], vs[1], buff.v[2], vs[2], buff.v[3] },
            };
            buff = {
                { buff.t[3], ts[3], buff.t[4], ts[4], buff.t[5], ts[5], buff.t[6] },
                { buff.v[3], vs[3], buff.v[4], vs[4], buff.v[5], vs[5], buff.v[6] },
            };
        }
        _generate_splines(eval_cb, add_sample, buff0, threshold);
    }
    _generate_splines(eval_cb, add_sample, buff, threshold);
}

template<typename EvalCB, typename AddSample>
static __attribute__((always_inline)) inline
void generate_splines(EvalCB &eval_cb, AddSample &add_sample, double len,
                      double threshold)
{
    bb_debug("generate_splines: len=%f\n", len);
    SplineBuffer buff;
    for (int i = 0; i < 7; i++)
        buff.t[i] = len * i / 6;
    memcpy(buff.v, eval_cb(7, buff.t), sizeof(double) * 7);
    _generate_splines(eval_cb, add_sample, buff, threshold);
}

static inline cubic_spline_t appoximate_spline(double v[5])
{
    double v0 = v[0];
    double v1 = v[1] * (2.0 / 3) + v[2] * (1.0 / 3);
    double v2 = v[3] * (2.0 / 3) + v[2] * (1.0 / 3);
    double v3 = v[4];
    // clamp v1 and v2 so that the numbers won't go too crazy
    if (v3 >= v0) {
        v1 = std::max(v0, std::min(v1, v3));
        v2 = std::max(v0, std::min(v2, v3));
    }
    else {
        v1 = std::max(v3, std::min(v1, v0));
        v2 = std::max(v3, std::min(v2, v0));
    }
    return spline_from_values(v0, v1, v2, v3);
}

template<typename RFSOCBackend>
static __attribute__((always_inline)) inline
void generate_channel_tonedata(RFSOCBackend *rb, ToneChannel &channel,
                               int64_t total_cycle)
{
    auto &tone_buffer = rb->tone_buffer;
    auto _add_tone_data = rb->generator->__pyx_vtab->add_tone_data;

    bool first_output = true;
    auto add_tone_data = [&] (int64_t cycles, cubic_spline_t freq, cubic_spline_t amp,
                              cubic_spline_t phase, bool sync, bool ff) {
        bb_debug("outputting tone data: chn=%d, cycles=%" PRId64 ", sync=%d, ff=%d\n",
                 channel.chn, cycles, sync, ff);
        if (_add_tone_data(rb->generator, channel.chn >> 1, channel.chn & 1,
                           cycles, freq, amp, phase,
                           { first_output, sync, ff }) < 0)
            throw 0;
        first_output = false;
    };
    assert(!tone_buffer.params[0].empty());
    assert(!tone_buffer.params[1].empty());
    assert(!tone_buffer.params[2].empty());

    bb_debug("Start outputting tone data for channel %d\n", channel.chn);

    int64_t cur_cycle = 0;

    int64_t freq_cycle = 0;
    int freq_idx = 0;
    auto freq_action = tone_buffer.params[(int)ToneFreq][freq_idx];
    int64_t freq_end_cycle = freq_cycle + freq_action.cycle_len;

    int64_t phase_cycle = 0;
    int phase_idx = 0;
    auto phase_action = tone_buffer.params[(int)TonePhase][phase_idx];
    int64_t phase_end_cycle = phase_cycle + phase_action.cycle_len;

    int64_t amp_cycle = 0;
    int amp_idx = 0;
    auto amp_action = tone_buffer.params[(int)ToneAmp][amp_idx];
    int64_t amp_end_cycle = amp_cycle + amp_action.cycle_len;

    int64_t ff_cycle = 0;
    int ff_idx = 0;
    auto ff_action = tone_buffer.ff[ff_idx];
    int64_t ff_end_cycle = ff_cycle + ff_action.cycle_len;

    while (true) {
        bool sync = false;

        // First figure out if we are starting a new action
        // and how long the current/new action last.
        if (freq_cycle == cur_cycle)
            sync |= freq_action.sync;
        int64_t action_end_cycle = freq_end_cycle;
        if (phase_cycle == cur_cycle)
            sync |= phase_action.sync;
        action_end_cycle = std::min(action_end_cycle, phase_end_cycle);
        if (amp_cycle == cur_cycle)
            sync |= amp_action.sync;
        action_end_cycle = std::min(action_end_cycle, amp_end_cycle);
        if (ff_cycle == cur_cycle)
            sync |= ff_action.sync;
        action_end_cycle = std::min(action_end_cycle, ff_end_cycle);
        bb_debug("find continuous range [%" PRId64 ", %" PRId64 "] on channel %d\n",
                 cur_cycle, action_end_cycle, channel.chn);

        auto forward_freq = [&] {
            assert(freq_idx + 1 < tone_buffer.params[(int)ToneFreq].size());
            freq_cycle = freq_end_cycle;
            freq_idx += 1;
            freq_action = tone_buffer.params[(int)ToneFreq][freq_idx];
            freq_end_cycle = freq_cycle + freq_action.cycle_len;
        };
        auto forward_amp = [&] {
            assert(amp_idx + 1 < tone_buffer.params[(int)ToneAmp].size());
            amp_cycle = amp_end_cycle;
            amp_idx += 1;
            amp_action = tone_buffer.params[(int)ToneAmp][amp_idx];
            amp_end_cycle = amp_cycle + amp_action.cycle_len;
        };
        auto forward_phase = [&] {
            assert(phase_idx + 1 < tone_buffer.params[(int)TonePhase].size());
            phase_cycle = phase_end_cycle;
            phase_idx += 1;
            phase_action = tone_buffer.params[(int)TonePhase][phase_idx];
            phase_end_cycle = phase_cycle + phase_action.cycle_len;
        };
        auto forward_ff = [&] {
            assert(ff_idx + 1 < tone_buffer.ff.size());
            ff_cycle = ff_end_cycle;
            ff_idx += 1;
            ff_action = tone_buffer.ff[ff_idx];
            ff_end_cycle = ff_cycle + ff_action.cycle_len;
        };

        if (action_end_cycle >= cur_cycle + 4) {
            // There's enough space to output a full tone data.
            auto resample_action_spline = [&] (auto action, int64_t action_cycle) {
                auto t1 = double(cur_cycle - action_cycle) / action.cycle_len;
                auto t2 = double(action_end_cycle - action_cycle) / action.cycle_len;
                return spline_resample(action.spline, t1, t2);
            };

            bb_debug("continuous range long enough for normal output (channel %d)\n",
                     channel.chn);
            add_tone_data(action_end_cycle - cur_cycle,
                          resample_action_spline(freq_action, freq_cycle),
                          resample_action_spline(amp_action, amp_cycle),
                          resample_action_spline(phase_action, phase_cycle),
                          sync, ff_action.ff);
            cur_cycle = action_end_cycle;
        }
        else {
            // The last action is at least 8 cycles long and we eat up at most
            // 4 cycles from it to handle pending sync so we should have at least
            // 4 cycles if we are hitting the end.
            assert(action_end_cycle != total_cycle);
            assert(cur_cycle + 4 <= total_cycle);
            bb_debug("continuous range too short (channel %d)\n", channel.chn);

            auto eval_param = [&] (auto &param, int64_t cycle, int64_t cycle_start) {
                auto dt = cycle - cycle_start;
                auto len = param.cycle_len;
                if (len == 0) {
                    assert(dt == 0);
                    return param.spline.order0;
                }
                return spline_eval(param.spline, double(dt) / len);
            };

            // Now we don't have enough time to do a tone data
            // based on the segmentation given to us. We'll manually iterate over
            // the next 4 cycles and compute a 4 cycle tone data that approximate
            // the action we need the closest.

            // This is the frequency we should sync at.
            // We need to record this exactly.
            // It's even more important than the frequency we left the channel at
            // since getting this wrong could mean a huge phase shift.
            int sync_cycle = 0;
            double sync_freq = 0;
            if (sync)
                sync_freq = eval_param(freq_action, cur_cycle, freq_cycle);
            double freqs[5];
            while (true) {
                auto min_cycle = (int)std::max(freq_cycle - cur_cycle, int64_t(0));
                auto max_cycle = (int)std::min(freq_end_cycle - cur_cycle, int64_t(4));
                for (int cycle = min_cycle; cycle <= max_cycle; cycle++)
                    freqs[cycle] = eval_param(freq_action, cur_cycle + cycle,
                                              freq_cycle);
                if (freq_end_cycle >= cur_cycle + 4)
                    break;
                forward_freq();
                if (freq_action.sync) {
                    sync = true;
                    sync_cycle = int(freq_cycle - cur_cycle);
                    sync_freq = freq_action.spline.order0;
                }
            }
            action_end_cycle = freq_end_cycle;
            double phases[5];
            while (true) {
                auto min_cycle = (int)std::max(phase_cycle - cur_cycle, int64_t(0));
                auto max_cycle = (int)std::min(phase_end_cycle - cur_cycle, int64_t(4));
                for (int cycle = min_cycle; cycle <= max_cycle; cycle++)
                    phases[cycle] = eval_param(phase_action, cur_cycle + cycle,
                                               phase_cycle);
                if (phase_end_cycle >= cur_cycle + 4)
                    break;
                forward_phase();
                if (phase_action.sync && phase_cycle > cur_cycle + sync_cycle) {
                    sync = true;
                    sync_cycle = int(phase_cycle - cur_cycle);
                    sync_freq = freqs[sync_cycle];
                }
            }
            action_end_cycle = std::min(action_end_cycle, phase_end_cycle);
            double amps[5];
            while (true) {
                auto min_cycle = (int)std::max(amp_cycle - cur_cycle, int64_t(0));
                auto max_cycle = (int)std::min(amp_end_cycle - cur_cycle, int64_t(4));
                for (int cycle = min_cycle; cycle <= max_cycle; cycle++)
                    amps[cycle] = eval_param(amp_action, cur_cycle + cycle,
                                             amp_cycle);
                if (amp_end_cycle >= cur_cycle + 4)
                    break;
                forward_amp();
                if (amp_action.sync && amp_cycle > cur_cycle + sync_cycle) {
                    sync = true;
                    sync_cycle = int(amp_cycle - cur_cycle);
                    sync_freq = freqs[sync_cycle];
                }
            }
            action_end_cycle = std::min(action_end_cycle, amp_end_cycle);
            while (true) {
                if (ff_end_cycle >= cur_cycle + 4)
                    break;
                forward_ff();
                if (ff_action.sync && ff_cycle > cur_cycle + sync_cycle) {
                    sync = true;
                    sync_cycle = int(ff_cycle - cur_cycle);
                    sync_freq = freqs[sync_cycle];
                }
            }
            action_end_cycle = std::min(action_end_cycle, ff_end_cycle);

            bb_debug("freq: {%f, %f, %f, %f, %f}\n",
                     freqs[0], freqs[1], freqs[2], freqs[3], freqs[4]);
            bb_debug("amp: {%f, %f, %f, %f, %f}\n",
                     amps[0], amps[1], amps[2], amps[3], amps[4]);
            bb_debug("phase: {%f, %f, %f, %f, %f}\n",
                     phases[0], phases[1], phases[2], phases[3], phases[4]);
            bb_debug("cur_cycle=%" PRId64 ", end_cycle=%" PRId64 "\n",
                     cur_cycle, action_end_cycle);

            // We can only sync at the start of the tone data so the start frequency
            // must be the sync frequency.
            if (sync) {
                freqs[0] = sync_freq;
                bb_debug("sync at %f\n", freqs[0]);
            }
            else {
                bb_debug("no sync\n");
            }
            add_tone_data(4, appoximate_spline(freqs), appoximate_spline(amps),
                          appoximate_spline(phases), sync, ff_action.ff);
            cur_cycle += 4;
            if (cur_cycle != action_end_cycle) {
                // We've only outputted 4 cycles (instead of outputting
                // to the end of an action) so in general there may not be anything
                // to post-process. However, if we happen to be hitting the end
                // of an action on the 4 cycle mark, we need to do the post-processing
                // to maintain the invariance that we are not at the end
                // of the sequence.
                assert(cur_cycle < action_end_cycle);
                continue;
            }
        }
        if (action_end_cycle == total_cycle)
            break;
        if (action_end_cycle == freq_end_cycle)
            forward_freq();
        if (action_end_cycle == amp_end_cycle)
            forward_amp();
        if (action_end_cycle == phase_end_cycle)
            forward_phase();
        if (action_end_cycle == ff_end_cycle)
            forward_ff();
    }
}

template<typename RFSOCBackend>
static __attribute__((always_inline)) inline
void generate_tonedata(RFSOCBackend *rb, unsigned age, const RuntimeVTable vtable)
{
    bb_debug("generate_tonedata: start\n");
    auto seq = rb->__pyx_base.seq;
    for (size_t i = 0, nreloc = rb->bool_values.size(); i < nreloc; i++) {
        auto &[rtval, val] = rb->bool_values[i];
        py_object pyval(vtable.rt_eval((PyObject*)rtval, age));
        if (!pyval)
            reraise_reloc_error(rb, i, true);
        val = get_value_bool(pyval, [&] { reraise_reloc_error(rb, i, true); });
    }
    for (size_t i = 0, nreloc = rb->float_values.size(); i < nreloc; i++) {
        auto &[rtval, val] = rb->float_values[i];
        py_object pyval(vtable.rt_eval((PyObject*)rtval, age));
        if (!pyval)
            reraise_reloc_error(rb, i, false);
        val = get_value_f64(pyval, [&] { reraise_reloc_error(rb, i, false); });
    }
    auto &time_values = seq->__pyx_base.__pyx_base.seqinfo->time_mgr->time_values;
    auto reloc_action = [rb, &time_values] (const RFSOCAction &action,
                                            ToneParam param) {
        auto reloc = rb->relocations[action.reloc_id];
        if (reloc.cond_idx != -1)
            action.cond = rb->bool_values[reloc.cond_idx].second;
        // No need to do anything else if we hit a disabled action.
        if (!action.cond)
            return;
        if (reloc.time_idx != -1)
            action.seq_time = time_values[reloc.time_idx];
        if (reloc.val_idx != -1) {
            if (param == ToneFF) {
                action.bool_value = rb->bool_values[reloc.val_idx].second;
            }
            else {
                action.float_value = rb->float_values[reloc.val_idx].second;
            }
        }
    };

    bool eval_status = !rb->eval_status;
    rb->eval_status = eval_status;
    auto ramp_buff = rb->ramp_buffer;

    constexpr double spline_threshold[3] = {
        [ToneFreq] = 10,
        [TonePhase] = 1e-3,
        [ToneAmp] = 1e-3,
    };

    // Add extra cycles to be able to handle the requirement of minimum 4 cycles.
    auto total_cycle = seq_time_to_cycle(rb->__pyx_base.seq->total_time) + 8;
    for (auto &channel: rb->channels.channels) {
        ScopeExit cleanup([&] {
            rb->tone_buffer.clear();
        });
        for (int param = 0; param < _NumToneParam; param++) {
            auto &actions = channel.actions[param];
            bb_debug("processing tone channel: %d, %s, nactions=%zd\n",
                     channel.chn, param_name(param), actions.size());
            if (actions.size() == 1) {
                auto &a = actions[0];
                if (a.reloc_id >= 0) {
                    a.eval_status = eval_status;
                    reloc_action(a, (ToneParam)param);
                }
            }
            else {
                std::sort(actions.begin(), actions.end(),
                          [&] (const auto &a1, const auto &a2) {
                              if (a1.reloc_id >= 0 && a1.eval_status != eval_status) {
                                  a1.eval_status = eval_status;
                                  reloc_action(a1, (ToneParam)param);
                              }
                              if (a2.reloc_id >= 0 && a2.eval_status != eval_status) {
                                  a2.eval_status = eval_status;
                                  reloc_action(a2, (ToneParam)param);
                              }
                              // Move disabled actions to the end
                              if (a1.cond != a2.cond)
                                  return int(a1.cond) > int(a2.cond);
                              // Sort by time
                              if (a1.seq_time != a2.seq_time)
                                  return a1.seq_time < a2.seq_time;
                              // Sometimes time points with different tid needs
                              // to be sorted by tid to get the output correct.
                              if (a1.tid != a2.tid)
                                  return a1.tid < a2.tid;
                              // End action technically happens
                              // just before the time point and must be sorted
                              // to be before the start action.
                              return int(a1.is_end) > int(a2.is_end);
                              // The frontend/shared finalization code only allow
                              // a single action on the same channel at the same time
                              // so there shouldn't be any ambiguity.
                          });
            }
            int64_t cur_cycle = 0;
            bool sync = false;
            if (param == ToneFF) {
                bool ff = false;
                auto &ff_action = rb->tone_buffer.ff;
                assert(ff_action.empty());
                for (auto &action: actions) {
                    if (!action.cond) {
                        bb_debug("found disabled %s action, finishing\n",
                                 param_name(param));
                        break;
                    }
                    // Nothing changed.
                    if (ff == action.bool_value && (sync || !action.sync)) {
                        bb_debug("skipping %s action: @%" PRId64 ", ff=%d\n",
                                 param_name(param),
                                 seq_time_to_cycle(action.seq_time), ff);
                        continue;
                    }
                    auto new_cycle = seq_time_to_cycle(action.seq_time);
                    if (new_cycle != cur_cycle) {
                        bb_debug("adding %s action: [%" PRId64 ", %" PRId64 "], "
                                 "cycle_len=%" PRId64 ", sync=%d, ff=%d\n",
                                 param_name(param), cur_cycle, new_cycle,
                                 new_cycle - cur_cycle, sync, ff);
                        assert(new_cycle > cur_cycle);
                        ff_action.push_back({ new_cycle - cur_cycle, sync, ff });
                        sync = false;
                        cur_cycle = new_cycle;
                    }
                    ff = action.bool_value;
                    sync |= action.sync;
                    bb_debug("%s status: @%" PRId64 ", sync=%d, ff=%d\n",
                             param_name(param), cur_cycle, sync, ff);
                }
                bb_debug("adding last %s action: [%" PRId64 ", %" PRId64 "], "
                         "cycle_len=%" PRId64 ", sync=%d, ff=%d\n", param_name(param),
                         cur_cycle, total_cycle, total_cycle - cur_cycle, sync, ff);
                assert(total_cycle > cur_cycle);
                ff_action.push_back({ total_cycle - cur_cycle, sync, ff });
                continue;
            }
            auto &param_action = rb->tone_buffer.params[param];
            assert(param_action.empty());
            double val = 0;
            double value_scale = 1;
            if (param == TonePhase)
                value_scale = 2 * M_PI; // tone data wants rad as phase unit.
            for (auto &action: actions) {
                if (!action.cond) {
                    bb_debug("found disabled %s action, finishing\n",
                             param_name(param));
                    break;
                }
                if (!action.isramp && val == action.float_value &&
                    (sync || !action.sync)) {
                    bb_debug("skipping %s action: @%" PRId64 ", val=%f\n",
                             param_name(param),
                             seq_time_to_cycle(action.seq_time), val);
                    continue;
                }
                auto action_seq_time = action.seq_time;
                auto new_cycle = seq_time_to_cycle(action_seq_time);
                if (new_cycle != cur_cycle) {
                    assert(new_cycle > cur_cycle);
                    bb_debug("adding %s action: [%" PRId64 ", %" PRId64 "], "
                             "cycle_len=%" PRId64 ", sync=%d, val=%f\n",
                             param_name(param), cur_cycle, new_cycle,
                             new_cycle - cur_cycle, sync, val);
                    param_action.push_back({ new_cycle - cur_cycle, sync,
                            spline_from_static(val * value_scale) });
                    sync = false;
                    cur_cycle = new_cycle;
                }
                else if (sync && !action.sync && param == ToneFreq) {
                    // we must not change the sync frequency.
                    // insert a zero length pulse
                    bb_debug("adding 0-length freq sync action: @%" PRId64 ", val=%f\n",
                             cur_cycle, val);
                    param_action.push_back({ 0, true,
                            spline_from_static(val * value_scale) });
                    sync = false;
                }
                sync |= action.sync;
                if (!action.isramp) {
                    val = action.float_value;
                    bb_debug("%s status: @%" PRId64 ", sync=%d, val=%f\n",
                             param_name(param), cur_cycle, sync, val);
                    continue;
                }
                auto len = action.float_value;
                auto ramp_func = action.ramp;
                bb_debug("processing ramp on %s: @%" PRId64 ", "
                         "sync=%d, len=%f, func=%p\n",
                         param_name(param), cur_cycle, sync, len, ramp_func);
                double sp_time;
                int64_t sp_seq_time;
                int64_t sp_cycle;
                double sp_cycle_time;
                auto update_sp_time = [&] (double t) {
                    static constexpr double time_scale = 1e12;
                    sp_time = t;
                    sp_seq_time = action_seq_time + int64_t(t * time_scale + 0.5);
                    sp_cycle = seq_time_to_cycle(sp_seq_time);
                    auto sp_cycle_seq_time = cycle_to_seq_time(sp_cycle);
                    sp_cycle_time = (sp_cycle_seq_time - action_seq_time) / time_scale;
                };
                update_sp_time(0);
                auto add_spline = [&] (double t2, cubic_spline_t sp) {
                    auto t1 = sp_time;
                    auto resample_t1 = (sp_cycle_time - t1) / (t2 - t1);
                    auto cycle1 = sp_cycle;
                    update_sp_time(t2);
                    auto resample_t2 = (sp_cycle_time - t1) / (t2 - t1);
                    auto cycle2 = sp_cycle;
                    bb_debug("adding %s spline: [%" PRId64 ", %" PRId64 "], "
                             "cycle_len=%" PRId64 ", sync=%d, "
                             "val=spline(%f, %f, %f, %f)\n",
                             param_name(param), cycle1, cycle2, cycle2 - cycle1, sync,
                             sp.order0, sp.order1, sp.order2, sp.order3);
                    assert(t2 >= t1);
                    param_action.push_back({ cycle2 - cycle1, sync,
                            spline_resample(sp, resample_t1, resample_t2)
                        });
                    sync = false;
                };
                if (cubic_spline_t sp; vtable.ramp_get_cubic_spline(ramp_func, &sp)) {
                    bb_debug("found SeqCubicSpline on %s spline: "
                             "old cycle:%" PRId64 "\n", param_name(param), cur_cycle);
                    val = sp.order0 + sp.order1 + sp.order2 + sp.order3;
                    if (value_scale != 1) {
                        sp.order0 *= value_scale;
                        sp.order1 *= value_scale;
                        sp.order2 *= value_scale;
                        sp.order3 *= value_scale;
                    }
                    add_spline(len, sp);
                    cur_cycle = sp_cycle;
                    bb_debug("found SeqCubicSpline on %s spline: "
                             "new cycle:%" PRId64 "\n", param_name(param), cur_cycle);
                    continue;
                }
                py_object py_oldval(pyfloat_from_double(val));
                py_object py_len(pyfloat_from_double(len));
                bb_reraise_and_throw_if(!py_oldval || !py_len, action_key(action.aid));
                double *input;
                double *output;
                int n = vtable.rampbuffer_eval_segments(
                    (PyObject*)ramp_buff, ramp_func, py_len, py_oldval,
                    &input, &output);
                bb_reraise_and_throw_if(n < 0, action_key(action.aid));
                auto add_sample = [&] (double t2, double v0, double v1,
                                       double v2, double v3) {
                    auto sp = spline_from_values(v0 * value_scale, v1 * value_scale,
                                                 v2 * value_scale, v3 * value_scale);
                    add_spline(t2, sp);
                    val = v3;
                };
                if (n > 0) {
                    assert(input[0] == 0);
                    bb_debug("Use ramp function provided segments on %s spline: "
                             "old cycle:%" PRId64 "\n", param_name(param), cur_cycle);
                    for (int i = 0; i < n - 1; i += 3)
                        add_sample(input[i + 3], output[i], output[i + 1],
                                   output[i + 2], output[i + 3]);
                    cur_cycle = sp_cycle;
                    bb_debug("Use ramp function provided segments on %s spline: "
                             "new cycle:%" PRId64 "\n", param_name(param), cur_cycle);
                    continue;
                }
                auto eval_ramp = [&] (int sz, double *inputs) {
                    auto input_buff =
                        vtable.rampbuffer_alloc_input((PyObject*)ramp_buff, sz);
                    bb_reraise_and_throw_if(!input_buff, action_key(action.aid));
                    memcpy(input_buff, inputs, sizeof(double) * sz);
                    auto output_buff =
                        vtable.rampbuffer_eval((PyObject*)ramp_buff, ramp_func,
                                               py_len, py_oldval);
                    bb_reraise_and_throw_if(!output_buff, action_key(action.aid));
                    return output_buff;
                };
                bb_debug("Use adaptive segments on %s spline: "
                         "old cycle:%" PRId64 "\n", param_name(param), cur_cycle);
                generate_splines(eval_ramp, add_sample, len,
                                 spline_threshold[param]);
                cur_cycle = sp_cycle;
                bb_debug("Use adaptive segments on %s spline: "
                         "new cycle:%" PRId64 "\n", param_name(param), cur_cycle);
            }
            param_action.push_back({ total_cycle - cur_cycle, sync,
                    spline_from_static(val * value_scale) });
        }
        generate_channel_tonedata(rb, channel, total_cycle);
    }
    bb_debug("generate_tonedata: finish\n");
}

}
