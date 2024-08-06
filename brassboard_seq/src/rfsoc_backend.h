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
    bool operator==(const cubic_spline_t &other) const
    {
        return (order0 == other.order0) && (order1 == other.order1) &&
            (order2 == other.order2) && (order3 == other.order3);
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
    void ensure_both_tones();
};

struct DDSParamAction {
    int64_t cycle_len: 63;
    bool sync: 1;
    cubic_spline_t spline;
};
struct DDSFFAction {
    int64_t cycle_len: 62;
    bool sync: 1;
    bool ff: 1;
};

struct ToneBuffer {
    std::vector<DDSParamAction> params[3];
    std::vector<DDSFFAction> ff;
    void clear()
    {
        for (auto &param: params)
            param.clear();
        ff.clear();
    }
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

}

#endif
