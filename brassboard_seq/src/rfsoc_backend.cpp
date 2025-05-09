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

#include "rfsoc_backend.h"

#include <bitset>
#include <cctype>

namespace brassboard_seq::rfsoc_backend {

using rtval::RuntimeValue;
using event_time::EventTime;

static inline int64_t seq_time_to_cycle(int64_t time)
{
    // Each cycle is 1 / 1ps / 409.6MHz sequence time unit
    // or 78125/32 sequence unit.
    constexpr auto numerator = 78125;
    constexpr auto denominator = 32;
    auto cycle_whole = time / numerator * denominator;
    auto cycle_frac = ((time % numerator) * denominator + numerator / 2) / numerator;
    return cycle_whole + cycle_frac;
}

static inline int64_t cycle_to_seq_time(int64_t cycle)
{
    constexpr auto numerator = 32;
    constexpr auto denominator = 78125;
    auto time_whole = cycle / numerator * denominator;
    auto time_frac = ((cycle % numerator) * denominator + numerator / 2) / numerator;
    return time_whole + time_frac;
}

__attribute__((visibility("protected")))
void ChannelsInfo::ensure_unused_tones(bool all)
{
    // For now, do not generate RFSoC data if there's no output.
    // This may be a problem if some of the sequences in a scan contains RFSoC outputs
    // while others don't. The artiq integration code would need to handle this case.
    if (channels.empty())
        return;
    // Ensuring both tone being availabe seems to make a difference sometimes.
    // (Could be due to pulse compiler bugs)
    std::bitset<64> tone_used;
    for (auto channel: channels)
        tone_used.set(channel.chn);
    for (int i = 0; i < 32; i++) {
        bool tone0 = tone_used.test(i * 2);
        bool tone1 = tone_used.test(i * 2 + 1);
        if ((tone0 || all) && !tone1) {
            channels.push_back({ i * 2 + 1 });
        }
        if (!tone0 && (tone1 || all)) {
            channels.push_back({ i * 2 });
        }
    }
}

static int parse_pos_int(const std::string_view &s, py::tuple path, int max)
{
    if (!std::ranges::all_of(s, [] (auto c) { return std::isdigit(c); }))
        config::raise_invalid_channel(path);
    int n;
    if (std::from_chars(s.data(), s.data() + s.size(), n).ec != std::errc{} || n > max)
        config::raise_invalid_channel(path);
    return n;
}

__attribute__((visibility("protected")))
void ChannelsInfo::collect_channel(py::ptr<seq::Seq> seq, py::str prefix)
{
    // Channel name format: <prefix>/dds<chn>/<tone>/<param>
    std::map<int,int> chn_idx_map;
    for (auto [idx, path]: py::list_iter<py::tuple>(seq->seqinfo->channel_paths)) {
        if (path.get<py::str>(0).compare(prefix) != 0)
            continue;
        if (path.size() != 4)
            config::raise_invalid_channel(path);
        auto ddspath = path.get<py::str>(1).utf8_view();
        if (!ddspath.starts_with("dds"))
            config::raise_invalid_channel(path);
        ddspath.remove_prefix(3);
        auto chn = ((parse_pos_int(ddspath, path, 31) << 1) |
                    parse_pos_int(path.get<py::str>(2).utf8_view(), path, 1));
        int chn_idx;
        if (auto it = chn_idx_map.find(chn); it == chn_idx_map.end()) {
            chn_idx = (int)channels.size();
            channels.push_back({ chn });
            chn_idx_map[chn] = chn_idx;
        }
        else {
            chn_idx = it->second;
        }
        auto param = path.get<py::str>(3);
        if (param.compare_ascii("freq") == 0) {
            chn_map.insert({(int)idx, {chn_idx, ToneFreq}});
        }
        else if (param.compare_ascii("phase") == 0) {
            chn_map.insert({(int)idx, {chn_idx, TonePhase}});
        }
        else if (param.compare_ascii("amp") == 0) {
            chn_map.insert({(int)idx, {chn_idx, ToneAmp}});
        }
        else if (param.compare_ascii("ff") == 0) {
            chn_map.insert({(int)idx, {chn_idx, ToneFF}});
        }
        else {
            config::raise_invalid_channel(path);
        }
    }
}

__attribute__((visibility("internal")))
void RFSOCBackend::Data::set_dds_delay(int dds, double delay)
{
    if (delay < 0)
        py_throw_format(PyExc_ValueError, "DDS time offset %S cannot be negative.",
                        py::new_float(delay));
    if (delay > 0.1)
        py_throw_format(PyExc_ValueError, "DDS time offset %S cannot be more than 100ms.",
                        py::new_float(delay));
    channels.set_dds_delay(dds, event_time::round_time_f64(delay));
}

static inline bool parse_action_kws(py::dict kws, int aid)
{
    assert(kws != Py_None);
    if (!kws)
        return false;
    bool sync = false;
    for (auto [key, value]: py::dict_iter<PyObject,py::str>(kws)) {
        if (key.compare_ascii("sync") == 0) {
            sync = value.as_bool(action_key(aid));
            continue;
        }
        bb_throw_format(PyExc_ValueError, action_key(aid),
                        "Invalid output keyword argument %S", kws);
    }
    return sync;
}

__attribute__((visibility("internal")))
void RFSOCBackend::Data::finalize(CompiledSeq &cseq)
{
    if (cseq.basic_cseqs.size() != 1)
        py_throw_format(PyExc_ValueError, "Branch not yet supported in rfsoc backend");
    channels.collect_channel(seq, prefix);

    ValueIndexer<bool> bool_values;
    ValueIndexer<double> float_values;
    py::list event_times = seq->seqinfo->time_mgr->event_times;

    channels.ensure_unused_tones(use_all_channels);

    for (auto [seq_chn, value]: channels.chn_map) {
        auto [chn_idx, param] = value;
        auto is_ff = param == ToneFF;
        auto &channel = channels.channels[chn_idx];
        auto &rfsoc_actions = channel.actions[(int)param];
        for (auto action: cseq.basic_cseqs[0]->chn_actions[seq_chn]->actions) {
            auto sync = parse_action_kws(action->kws, action->aid);
            py::ptr value = action->value;
            auto is_ramp = action::isramp(value);
            if (is_ff && is_ramp)
                bb_throw_format(PyExc_ValueError, action_key(action->aid),
                                "Feed forward control cannot be ramped");
            py::ptr cond = action->cond;
            if (cond == Py_False)
                continue;
            bool cond_need_reloc = rtval::is_rtval(cond);
            assert(cond_need_reloc || cond == Py_True);
            int cond_idx = cond_need_reloc ? bool_values.get_id(cond) : -1;
            auto add_action = [&] (py::ptr<> value, int tid, bool sync, bool is_ramp,
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

                auto event_time = event_times.get<EventTime>(tid);
                if (event_time->data.is_static()) {
                    rfsoc_action.seq_time = event_time->data._get_static();
                }
                else {
                    needs_reloc = true;
                    reloc.time_idx = tid;
                }
                if (is_ramp) {
                    rfsoc_action.ramp = value;
                    auto len = py::ptr(action->length);
                    if (rtval::is_rtval(len)) {
                        needs_reloc = true;
                        reloc.val_idx = float_values.get_id(len);
                    }
                    else {
                        rfsoc_action.float_value = len.as_float(action_key(action->aid));
                    }
                }
                else if (rtval::is_rtval(value)) {
                    needs_reloc = true;
                    if (is_ff) {
                        reloc.val_idx = bool_values.get_id(value);
                    }
                    else {
                        reloc.val_idx = float_values.get_id(value);
                    }
                }
                else if (is_ff) {
                    rfsoc_action.bool_value = value.as_bool(action_key(action->aid));
                }
                else {
                    rfsoc_action.float_value = value.as_float(action_key(action->aid));
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
            if (action->is_pulse || is_ramp) {
                add_action(action->end_val, action->end_tid, false, false, true);
            }
        }
    }

    this->bool_values = std::move(bool_values.values);
    this->float_values = std::move(float_values.values);
}

static constexpr double min_spline_time = 150e-9;

struct SplineBuffer {
    double t[7];
    double v[7];

    bool is_accurate_enough(double threshold) const
    {
        if (t[3] - t[0] <= min_spline_time)
            return true;
        auto sp = cubic_spline::from_values(v[0], v[2], v[4], v[6]);
        if (abs(sp.eval(1.0 / 6) - v[1]) > threshold)
            return false;
        if (abs(sp.eval(1.0 / 2) - v[3]) > threshold)
            return false;
        if (abs(sp.eval(5.0 / 6) - v[5]) > threshold)
            return false;
        return true;
    }
};

static __attribute__((flatten))
void _generate_splines(auto &eval_cb, auto &add_sample, SplineBuffer &buff,
                       double threshold)
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
            double vs[6];
            for (int i = 0; i < 6; i++) {
                double t = (buff.t[i] + buff.t[i + 1]) / 2;
                ts[i] = t;
                vs[i] = eval_cb(t);
            }
            bb_debug("evaluate on {%f, %f, %f, %f, %f, %f}\n",
                     ts[0], ts[1], ts[2], ts[3], ts[4], ts[5]);
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

static __attribute__((always_inline)) inline
void generate_splines(auto &eval_cb, auto &add_sample, double len, double threshold)
{
    bb_debug("generate_splines: len=%f\n", len);
    SplineBuffer buff;
    for (int i = 0; i < 7; i++) {
        auto t = len * i / 6;
        buff.t[i] = t;
        buff.v[i] = eval_cb(t);
    }
    _generate_splines(eval_cb, add_sample, buff, threshold);
}

__attribute__((visibility("internal")))
void RFSOCBackend::Data::runtime_finalize(CompiledSeq &cseq, unsigned age)
{
    bb_debug("rfsoc_runtime_finalize: start\n");
    for (auto [dds, delay]: py::dict_iter<RuntimeValue>(rt_dds_delay)) {
        rt_eval_throw(delay, age);
        set_dds_delay(dds.as_int(), rtval_cache(delay).template get<double>());
    }

    for (size_t i = 0, nreloc = bool_values.size(); i < nreloc; i++) {
        auto &[rtval, val] = bool_values[i];
        val = !rtval::rtval_cache(rtval).is_zero();
    }
    for (size_t i = 0, nreloc = float_values.size(); i < nreloc; i++) {
        auto &[rtval, val] = float_values[i];
        val = rtval::rtval_cache(rtval).template get<double>();
    }
    auto &time_values = seq->seqinfo->time_mgr->time_values;
    auto reloc_action = [this, &time_values] (const RFSOCAction &action,
                                              ToneParam param) {
        auto reloc = relocations[action.reloc_id];
        if (reloc.cond_idx != -1)
            action.cond = bool_values[reloc.cond_idx].second;
        // No need to do anything else if we hit a disabled action.
        if (!action.cond)
            return;
        if (reloc.time_idx != -1)
            action.seq_time = time_values[reloc.time_idx];
        if (reloc.val_idx != -1) {
            if (param == ToneFF) {
                action.bool_value = bool_values[reloc.val_idx].second;
            }
            else {
                action.float_value = float_values[reloc.val_idx].second;
            }
        }
    };

    bool eval_status = !this->eval_status;
    this->eval_status = eval_status;

    constexpr double spline_threshold[3] = {
        [ToneFreq] = 10,
        [TonePhase] = 1e-3,
        [ToneAmp] = 1e-3,
    };

    int64_t max_delay = 0;
    for (auto [dds, delay]: channels.dds_delay)
        max_delay = std::max(max_delay, delay);

    auto reloc_and_cmp_action = [&] (const auto &a1, const auto &a2, auto param) {
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
    };

    auto reloc_sort_actions = [&] (auto &actions, auto param) {
        if (actions.size() == 1) {
            auto &a = actions[0];
            if (a.reloc_id >= 0) {
                a.eval_status = eval_status;
                reloc_action(a, (ToneParam)param);
            }
        }
        else {
            std::ranges::sort(actions, [&] (const auto &a1, const auto &a2) {
                return reloc_and_cmp_action(a1, a2, param);
            });
        }
    };

    auto gen = generator->gen.get();
    gen->start();

    // Add extra cycles to be able to handle the requirement of minimum 4 cycles.
    auto total_cycle = seq_time_to_cycle(cseq.basic_cseqs[0]->total_time + max_delay) + 8;
    for (auto &channel: channels.channels) {
        ScopeExit cleanup([&] {
            tone_buffer.clear();
        });
        int64_t dds_delay = 0;
        if (auto it = channels.dds_delay.find(channel.chn >> 1);
            it != channels.dds_delay.end())
            dds_delay = it->second;
        auto sync_mgr = tone_buffer.syncs;
        {
            auto &actions = channel.actions[ToneFF];
            bb_debug("processing tone channel: %d, ff, nactions=%zd\n",
                     channel.chn, actions.size());
            reloc_sort_actions(actions, ToneFF);
            int64_t cur_cycle = 0;
            bool ff = false;
            auto &ff_action = tone_buffer.ff;
            assert(ff_action.empty());
            for (auto &action: actions) {
                if (!action.cond) {
                    bb_debug("found disabled ff action, finishing\n");
                    break;
                }
                auto action_seq_time = action.seq_time + dds_delay;
                auto new_cycle = seq_time_to_cycle(action_seq_time);
                if (action.sync)
                    sync_mgr.add(action_seq_time, new_cycle, action.tid, ToneFF);
                // Nothing changed.
                if (ff == action.bool_value) {
                    bb_debug("skipping ff action: @%" PRId64 ", ff=%d\n",
                             new_cycle, ff);
                    continue;
                }
                if (new_cycle != cur_cycle) {
                    bb_debug("adding ff action: [%" PRId64 ", %" PRId64 "], "
                             "cycle_len=%" PRId64 ", ff=%d\n",
                             cur_cycle, new_cycle, new_cycle - cur_cycle, ff);
                    assert(new_cycle > cur_cycle);
                    ff_action.push_back({ new_cycle - cur_cycle, ff });
                    cur_cycle = new_cycle;
                }
                ff = action.bool_value;
                bb_debug("ff status: @%" PRId64 ", ff=%d\n", cur_cycle, ff);
            }
            bb_debug("adding last ff action: [%" PRId64 ", %" PRId64 "], "
                     "cycle_len=%" PRId64 ", ff=%d\n", cur_cycle,
                     total_cycle, total_cycle - cur_cycle, ff);
            assert(total_cycle > cur_cycle);
            ff_action.push_back({ total_cycle - cur_cycle, ff });
        }
        for (auto param: { ToneAmp, TonePhase, ToneFreq }) {
            auto &actions = channel.actions[param];
            sync_mgr.init_output(param);
            bb_debug("processing tone channel: %d, %s, nactions=%zd\n",
                     channel.chn, param_name(param), actions.size());
            reloc_sort_actions(actions, param);
            int64_t cur_cycle = 0;
            auto &param_action = tone_buffer.params[param];
            assert(param_action.empty());
            double val = 0;
            int prev_tid = -1;
            for (auto &action: actions) {
                if (!action.cond) {
                    bb_debug("found disabled %s action, finishing\n",
                             param_name(param));
                    break;
                }
                auto action_seq_time = action.seq_time + dds_delay;
                auto new_cycle = seq_time_to_cycle(action_seq_time);
                if (action.sync)
                    sync_mgr.add(action_seq_time, new_cycle, action.tid, param);
                if (!action.isramp && val == action.float_value) {
                    bb_debug("skipping %s action: @%" PRId64 ", val=%f\n",
                             param_name(param), new_cycle, val);
                    continue;
                }
                sync_mgr.add_action(param_action, cur_cycle, new_cycle,
                                    cubic_spline::from_static(val), action_seq_time,
                                    prev_tid, param);
                cur_cycle = new_cycle;
                prev_tid = action.tid;
                if (!action.isramp) {
                    val = action.float_value;
                    bb_debug("%s status: @%" PRId64 ", val=%f\n",
                             param_name(param), cur_cycle, val);
                    continue;
                }
                auto len = action.float_value;
                auto ramp_func = (action::RampFunctionBase*)action.ramp;
                bb_debug("processing ramp on %s: @%" PRId64 ", len=%f, func=%p\n",
                         param_name(param), cur_cycle, len, ramp_func);
                double sp_time;
                int64_t sp_seq_time;
                int64_t sp_cycle;
                auto update_sp_time = [&] (double t) {
                    static constexpr double time_scale = event_time::time_scale;
                    sp_time = t;
                    sp_seq_time = action_seq_time + int64_t(t * time_scale + 0.5);
                    sp_cycle = seq_time_to_cycle(sp_seq_time);
                };
                update_sp_time(0);
                auto add_spline = [&] (double t2, cubic_spline sp) {
                    assert(t2 >= sp_time);
                    auto cycle1 = sp_cycle;
                    update_sp_time(t2);
                    auto cycle2 = sp_cycle;
                    // The spline may not actually start on the cycle.
                    // However, attempting to resample the spline results in
                    // more unique splines being created which seems to be overflowing
                    // the buffer on the hardware.
                    sync_mgr.add_action(param_action, cycle1, cycle2,
                                        sp, sp_seq_time, prev_tid, param);
                };
                if (auto py_spline =
                    py::cast<action::SeqCubicSpline,true>(ramp_func)) {
                    bb_debug("found SeqCubicSpline on %s spline: "
                             "old cycle:%" PRId64 "\n", param_name(param), cur_cycle);
                    cubic_spline sp{py_spline->data()->sp};
                    val = sp.order0 + sp.order1 + sp.order2 + sp.order3;
                    add_spline(len, sp);
                    cur_cycle = sp_cycle;
                    bb_debug("found SeqCubicSpline on %s spline: "
                             "new cycle:%" PRId64 "\n", param_name(param), cur_cycle);
                    continue;
                }
                auto add_sample = [&] (double t2, double v0, double v1,
                                       double v2, double v3) {
                    add_spline(t2, cubic_spline::from_values(v0, v1, v2, v3));
                    val = v3;
                };
                auto eval_ramp = [&] (double t) {
                    auto v = ramp_func->runtime_eval(t);
                    throw_py_error(v.err);
                    return v.val.f64_val;
                };
                py::ref<> pts;
                try {
                    pts = ramp_func->spline_segments(len, val);
                }
                catch (...) {
                    bb_rethrow(action_key(action.aid));
                }
                if (pts == Py_None) {
                    bb_debug("Use adaptive segments on %s spline: "
                             "old cycle:%" PRId64 "\n", param_name(param), cur_cycle);
                    generate_splines(eval_ramp, add_sample, len,
                                     spline_threshold[param]);
                    cur_cycle = sp_cycle;
                    bb_debug("Use adaptive segments on %s spline: "
                             "new cycle:%" PRId64 "\n", param_name(param), cur_cycle);
                    continue;
                }
                double prev_t = 0;
                double prev_v = eval_ramp(0);
                bb_debug("Use ramp function provided segments on %s spline: "
                         "old cycle:%" PRId64 "\n", param_name(param), cur_cycle);
                for (auto item: pts.generic_iter(action_key(action.aid))) {
                    double t = PyFloat_AsDouble(item.get());
                    if (!(t > prev_t)) [[unlikely]] {
                        if (!PyErr_Occurred()) {
                            if (t < 0) {
                                PyErr_Format(PyExc_ValueError,
                                             "Segment time cannot be negative");
                            }
                            else {
                                PyErr_Format(PyExc_ValueError,
                                             "Segment time point must "
                                             "monotonically increase");
                            }
                        }
                        bb_rethrow(action_key(action.aid));
                    }
                    auto t1 = t * (1.0 / 3.0) + prev_t * (2.0 / 3.0);
                    auto t2 = t * (2.0 / 3.0) + prev_t * (1.0 / 3.0);
                    auto t3 = t;
                    auto v0 = prev_v;
                    auto v1 = eval_ramp(t1);
                    auto v2 = eval_ramp(t2);
                    auto v3 = eval_ramp(t3);
                    add_sample(t3, v0, v1, v2, v3);
                    prev_t = t3;
                    prev_v = v3;
                }
                if (!(prev_t < len)) [[unlikely]]
                    bb_throw_format(PyExc_ValueError, action_key(action.aid),
                                    "Segment time point must not "
                                    "exceed action length.");
                auto t1 = len * (1.0 / 3.0) + prev_t * (2.0 / 3.0);
                auto t2 = len * (2.0 / 3.0) + prev_t * (1.0 / 3.0);
                auto t3 = len;
                auto v0 = prev_v;
                auto v1 = eval_ramp(t1);
                auto v2 = eval_ramp(t2);
                auto v3 = eval_ramp(t3);
                add_sample(t3, v0, v1, v2, v3);
                cur_cycle = sp_cycle;
                bb_debug("Use ramp function provided segments on %s spline: "
                         "new cycle:%" PRId64 "\n", param_name(param), cur_cycle);
            }
            sync_mgr.add_action(param_action, cur_cycle, total_cycle,
                                cubic_spline::from_static(val),
                                cycle_to_seq_time(total_cycle),
                                prev_tid, param);
        }
        gen->process_channel(tone_buffer, channel.chn, total_cycle);
    }
    gen->end();
    bb_debug("rfsoc_runtime_finalize: finish\n");
}

__attribute__((visibility("protected")))
PyTypeObject RFSOCBackend::Type = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "brassboard_seq.rfsoc_backend.RFSOCBackend",
    .tp_basicsize = sizeof(Backend) + sizeof(RFSOCBackend::Data),
    .tp_dealloc = py::tp_cxx_dealloc<true,RFSOCBackend>,
    .tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_HAVE_GC,
    .tp_traverse = traverse<>,
    .tp_clear = clear<>,
    .tp_methods = (
        py::meth_table<
        py::meth_fast<"set_dds_delay",[] (py::ptr<RFSOCBackend> self,
                                          PyObject *const *args, Py_ssize_t nargs) {
            py::check_num_arg("RFSOCBackend.set_dds_delay", nargs, 2, 2);
            auto dds = py::arg_cast<py::int_>(args[0], "dds");
            py::ptr delay = args[1];
            if (rtval::is_rtval(delay)) {
                self->data()->rt_dds_delay.set(dds, delay);
            }
            else {
                self->data()->set_dds_delay(dds.as_int(), delay.as_float());
            }
        }>>),
    .tp_getset = (py::getset_table<
                  py::getset_def<"use_all_channels",[] (py::ptr<RFSOCBackend> self) {
                      return py::new_bool(self->data()->use_all_channels);
                  },[] (py::ptr<RFSOCBackend> self, py::ptr<> _all_chns) {
                      self->data()->use_all_channels =
                          py::arg_cast<py::bool_,true>(_all_chns,
                                                       "use_all_channels").as_bool();
                  }>,
                  py::getset_def<"has_output",[] (py::ptr<RFSOCBackend> self) {
                      return py::new_bool(self->data()->channels.channels.size() != 0);
                  }>>),
    .tp_base = &Backend::Type,
    .tp_vectorcall = py::vectorfunc<[] (PyObject*, PyObject *const *args,
                                        ssize_t nargs, py::tuple kwnames) {
        py::check_num_arg("RFSOCBackend.__init__", nargs, 1, 1);
        py::check_no_kwnames("RFSOCBackend.__init__", kwnames);
        return alloc(py::arg_cast<RFSOCGenerator>(args[0], "generator"));
    }>
};

__attribute__((visibility("hidden")))
void init()
{
    throw_if(PyType_Ready(&RFSOCBackend::Type) < 0);
}

}
