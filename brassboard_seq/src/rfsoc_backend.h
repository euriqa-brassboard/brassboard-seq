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

#include "rfsoc.h"
#include "rfsoc_gen.h"

#include "backend.h"
#include "seq.h"

#include <algorithm>
#include <map>
#include <vector>
#include <utility>

namespace brassboard_seq::rfsoc_backend {

using namespace backend;
using namespace rfsoc;
using namespace rfsoc_gen;

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
    py::ptr<> ramp;
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

struct ChannelsInfo {
    std::vector<ToneChannel> channels;
    // map from sequence channel to tone channel index
    std::map<int,std::pair<int,ToneParam>> chn_map;
    std::map<int, int64_t> dds_delay;

    void collect_channel(py::ptr<seq::Seq> seq, py::str prefix);
    void set_dds_delay(int dds, int64_t delay)
    {
        dds_delay[dds] = delay;
    }
    void ensure_unused_tones(bool all);
};

struct RFSOCBackend : BackendBase::Base<RFSOCBackend> {
    struct Data final : BackendBase::Data {
        py::ref<RFSOCGenerator> generator;
        ChannelsInfo channels;
        std::vector<std::pair<void*,bool>> bool_values;
        std::vector<std::pair<void*,double>> float_values;
        std::vector<Relocation> relocations;
        bool eval_status{false};
        bool use_all_channels{false};
        ToneBuffer tone_buffer;

        py::dict_ref rt_dds_delay{py::new_dict()};

        Data(py::ptr<RFSOCGenerator> generator)
            : generator(generator.ref())
        {
        }

        void finalize(py::ptr<SeqCompiler>) override;
        void runtime_finalize(py::ptr<SeqCompiler>, unsigned) override;
        void set_dds_delay(int dds, double delay);
    };

    using fields = field_pack<BackendBase::fields,&Data::generator,&Data::rt_dds_delay>;
    static PyTypeObject Type;
};

void init();

}

#endif
