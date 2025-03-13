/*************************************************************************
 *   Copyright (c) 2024 - 2024 Yichao Yu <yyc1992@gmail.com>             *
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

#ifndef BRASSBOARD_SEQ_SRC_RFSOC_BACKEND_H
#define BRASSBOARD_SEQ_SRC_RFSOC_BACKEND_H

#include <algorithm>
#include <array>
#include <map>
#include <vector>
#include <utility>

#include <stdint.h>

#include <Python.h>

namespace brassboard_seq::rfsoc_backend {

struct cubic_spline_t {
    double order0;
    double order1;
    double order2;
    double order3;
    constexpr bool operator==(const cubic_spline_t &other) const
    {
        return (order0 == other.order0) && (order1 == other.order1) &&
            (order2 == other.order2) && (order3 == other.order3);
    }
    constexpr std::array<double,4> to_array() const
    {
        return { order0, order1, order2, order3 };
    }
};

struct output_flags_t {
    bool wait_trigger;
    bool sync;
    bool feedback_enable;
};

enum ToneParam {
    ToneFreq,
    TonePhase,
    ToneAmp,
    ToneFF,
    _NumToneParam
};

struct RFSOCAction {
    mutable bool cond;
    mutable bool eval_status;
    bool isramp;
    bool sync;
    int reloc_id;
    int aid;
    int tid: 31;
    bool is_end: 1;
    mutable int64_t seq_time;
    union {
        mutable double float_value;
        mutable bool bool_value;
    };
    PyObject *ramp;
};

struct Relocation {
    // If a particular relocation is not needed for this action,
    // the corresponding idx would be -1
    int cond_idx;
    int time_idx;
    int val_idx;
};

struct ToneChannel {
    ToneChannel(int chn)
        : chn(chn)
    {
    }

    int chn; // ddsnum << 1 | tone
    std::vector<RFSOCAction> actions[4];
};

struct ChannelInfo {
    std::vector<ToneChannel> channels;
    // map from sequence channel to tone channel index
    std::map<int,std::pair<int,ToneParam>> chn_map;
    std::map<int, int64_t> dds_delay;

    int add_tone_channel(int chn);
    void add_seq_channel(int seq_chn, int chn_idx, ToneParam param);
    void set_dds_delay(int dds, int64_t delay)
    {
        dds_delay[dds] = delay;
    }
    void ensure_unused_tones(bool all);
};

struct DDSParamAction {
    int64_t cycle_len: 63;
    bool sync: 1;
    cubic_spline_t spline;
};
struct DDSFFAction {
    int64_t cycle_len: 63;
    bool ff: 1;
};

static inline const char *param_name(ToneParam param)
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

struct SyncTimeMgr {
    struct SyncInfo {
        int64_t seq_time{};
        int tid{};
    };
    std::map<int64_t,SyncInfo> times;
    std::map<int64_t,SyncInfo>::iterator next_it;
    // invariant: if next_it.first < cur_seq_time then sync_freq is valid.
    double sync_freq;
    int64_t sync_freq_seq_time;
    bool sync_freq_match_tid;

    void clear()
    {
        times.clear();
        next_it = times.end();
    }

    void init_output(ToneParam param)
    {
        if (param == ToneFreq) {
            next_it = times.begin();
            sync_freq = 0;
            sync_freq_seq_time = -1;
            sync_freq_match_tid = false;
        }
    }

    void add(int64_t seq_time, int64_t cycle, int tid, ToneParam param)
    {
        bb_debug("Collected sync action from %s @%" PRId64 ", tid=%d\n",
                 param_name(param), cycle, tid);
        auto [it, inserted] = times.emplace(cycle, SyncInfo{ seq_time, tid });
        if (!inserted && (it->second.seq_time < seq_time ||
                          (it->second.seq_time == seq_time && it->second.tid < tid)))
            it->second = { seq_time, tid };
        if (param == ToneFreq) {
            if (next_it == times.end() || next_it->first > cycle) {
                assert(seq_time >= sync_freq_seq_time);
                next_it = it;
            }
        }
    }

    void add_action(std::vector<DDSParamAction> &actions, int64_t start_cycle,
                    int64_t end_cycle, cubic_spline_t sp,
                    int64_t end_seq_time, int tid, ToneParam param);
};

struct ToneBuffer {
    std::vector<DDSParamAction> params[3];
    std::vector<DDSFFAction> ff;
    SyncTimeMgr syncs;
    void clear()
    {
        for (auto &param: params)
            param.clear();
        ff.clear();
        syncs.clear();
    }
};

struct Generator {
    virtual void start() = 0;
    virtual void process_channel(ToneBuffer &tone_buffer, int chn,
                                 int64_t total_cycle) = 0;
    virtual void end() = 0;
    virtual ~Generator() = default;
};

static inline int64_t seq_time_to_cycle(long long time)
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

static __attribute__((always_inline)) inline
cubic_spline_t spline_from_static(double v0)
{
    return { v0, 0, 0, 0 };
}

static __attribute__((optimize("-ffast-math"),always_inline)) inline
cubic_spline_t spline_from_values(double v0, double v1, double v2, double v3)
{
    // v = o0 + o1 * t + o2 * t^2 + o3 * t^3

    // v0 = o0
    // v1 = o0 + o1 / 3 + o2 / 9 + o3 / 27
    // v2 = o0 + o1 * 2 / 3 + o2 * 4 / 9 + o3 * 8 / 27
    // v3 = o0 + o1 + o2 + o3

    // o0 = v0
    // o1 = -5.5 * v0 + 9 * v1 - 4.5 * v2 + v3
    // o2 = 9 * v0 - 22.5 * v1 + 18 * v2 - 4.5 * v3
    // o3 = -4.5 * v0 + 13.5 * v1 - 13.5 * v2 + 4.5 * v3

    return {
        v0,
        -5.5 * v0 + 9 * v1 - 4.5 * v2 + v3,
        9 * v0 - 22.5 * v1 + 18 * v2 - 4.5 * v3,
        -4.5 * v0 + 13.5 * v1 - 13.5 * v2 + 4.5 * v3,
    };
}

static __attribute__((optimize("-ffast-math"),always_inline)) inline
double spline_eval(cubic_spline_t spline, double t)
{
    return spline.order0 + (spline.order1 +
                            (spline.order2 + spline.order3 * t) * t) * t;
}

static __attribute__((optimize("-ffast-math"),always_inline)) inline
cubic_spline_t spline_resample(cubic_spline_t spline, double t1, double t2)
{
    double dt = t2 - t1;
    double dt2 = dt * dt;
    double dt3 = dt2 * dt;
    double o3_3 = 3 * spline.order3;
    return {
        spline.order0 + (spline.order1 +
                         (spline.order2 + spline.order3 * t1) * t1) * t1,
        dt * (spline.order1 + (2 * spline.order2 + o3_3 * t1) * t1),
        dt2 * (spline.order2 + o3_3 * t1),
        dt3 * spline.order3,
    };
}

static inline cubic_spline_t approximate_spline(double v[5])
{
    double v0 = v[0];
    double v1 = v[1] * (2.0 / 3) + v[2] * (1.0 / 3);
    double v2 = v[3] * (2.0 / 3) + v[2] * (1.0 / 3);
    double v3 = v[4];
    // clamp v1 and v2 so that the numbers won't go too crazy
    if (v3 >= v0) {
        v1 = std::clamp(v1, v0, v3);
        v2 = std::clamp(v2, v0, v3);
    }
    else {
        v1 = std::clamp(v1, v3, v0);
        v2 = std::clamp(v2, v3, v0);
    }
    return spline_from_values(v0, v1, v2, v3);
}

}

#endif
