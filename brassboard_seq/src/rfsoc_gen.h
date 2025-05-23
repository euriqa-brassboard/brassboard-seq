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

#ifndef BRASSBOARD_SEQ_SRC_RFSOC_GEN_H
#define BRASSBOARD_SEQ_SRC_RFSOC_GEN_H

#include "rfsoc.h"

#include <algorithm>
#include <map>
#include <vector>
#include <utility>

namespace brassboard_seq::rfsoc_gen {

using namespace rfsoc;

enum ToneParam {
    ToneFreq,
    TonePhase,
    ToneAmp,
    ToneFF,
    _NumToneParam
};

struct DDSParamAction {
    int64_t cycle_len: 63;
    bool sync: 1;
    cubic_spline spline;
    double eval(int64_t dt)
    {
        if (cycle_len == 0) {
            assert(dt == 0);
            return spline.order0;
        }
        return spline.eval(double(dt) / cycle_len);
    }
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
                    int64_t end_cycle, cubic_spline sp,
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

struct output_flags_t {
    bool wait_trigger;
    bool sync;
    bool feedback_enable;
};

struct Generator {
    virtual void start() = 0;
    virtual void process_channel(ToneBuffer &tone_buffer, int chn,
                                 int64_t total_cycle) = 0;
    virtual void end() = 0;
    virtual ~Generator() = default;
};

struct Jaqalv1Gen final: Generator {
    struct BoardGen {
        Jaqal_v1::ChannelGen channels[8];
        void clear();
        py::ref<> get_prefix() const;
        py::ref<> get_sequence() const;
        void end();
    };
    BoardGen boards[4]; // 4 * 8 physical channels

    void start() override;
    void process_channel(ToneBuffer &tone_buffer, int chn,
                         int64_t total_cycle) override;
    void add_tone_data(int chn, int64_t duration_cycles, cubic_spline freq,
                       cubic_spline amp, cubic_spline phase,
                       output_flags_t flags, int64_t cur_cycle);
    void end() override;
    py::ref<> get_prefix(int n) const;
    py::ref<> get_sequence(int n) const;
};

struct Jaqalv1_3Gen final: Generator {
    py::ref<> get_prefix(int n) const;
    py::ref<> get_sequence(int n) const;
private:
    struct ChnInfo {
        uint16_t board;
        uint8_t board_chn;
        uint8_t tone;
        ChnInfo(int chn)
            : board(chn / 16),
              board_chn((chn / 2) % 8),
              tone(chn % 2)
        {
        }
    };
    static inline int64_t limit_cycles(int64_t cur, int64_t end)
    {
        int64_t max_cycles = (int64_t(1) << 40) - 1;
        int64_t len = end - cur;
        if (len <= max_cycles)
            return end;
        if (len > max_cycles * 2)
            return cur + max_cycles;
        return cur + len / 2;
    }

    std::vector<TimedInst> board_insts[4];

    void process_freq(std::span<DDSParamAction> freq, std::span<DDSFFAction> ff,
                      ChnInfo chn, int64_t total_cycle);
    template<auto pulsef>
    void process_param(std::span<DDSParamAction> param, ChnInfo chn,
                       int64_t total_cycle, Jaqal_v1_3::ModType modtype);
    void process_frame(ChnInfo chn, int64_t total_cycle, Jaqal_v1_3::ModType modtype);
    void add_inst(const JaqalInst &inst, int board, int board_chn,
                  Jaqal_v1_3::ModType mod, int64_t cycle);
    void start() override;
    void process_channel(ToneBuffer &tone_buffer, int chn, int64_t total_cycle) override;
    void end() override;
};

struct RFSOCGenerator : PyObject {
    std::unique_ptr<Generator> gen;

    static PyTypeObject Type;
};

extern PyTypeObject &Jaqalv1Generator_Type;
extern PyTypeObject &Jaqalv1_3Generator_Type;

void init();

}

#endif
