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

}

#endif
