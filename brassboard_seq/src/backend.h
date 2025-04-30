/*************************************************************************
 *   Copyright (c) 2025 - 2025 Yichao Yu <yyc1992@gmail.com>             *
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

#ifndef BRASSBOARD_SEQ_SRC_BACKEND_H
#define BRASSBOARD_SEQ_SRC_BACKEND_H

#include "utils.h"
#include "action.h"
#include "seq.h"

#include <list>
#include <memory>
#include <vector>

namespace brassboard_seq::backend {

struct ChannelAction {
    std::vector<action::Action*> actions;
    py::ref<> start_value;
};

struct CompiledBasicSeq {
    int bseq_id;
    bool may_term;
    std::unique_ptr<ChannelAction*[]> chn_actions;
    int64_t total_time;
    std::vector<int> next_bseq;
};

struct CompiledSeq {
    int nchn;
    int nbseq;
    std::vector<CompiledBasicSeq*> basic_cseqs;
    PermAllocator<CompiledBasicSeq,16> basic_seq_alloc;
    PermAllocator<ChannelAction,16> chn_action_alloc;
    std::vector<std::vector<ChannelAction*>> all_chn_actions;
    void initialize(py::ptr<seq::Seq>);
    void populate_values(py::ptr<seq::Seq>);
    // Use std::vector<uint8_t> to pass in the status rather than std::vector<bool>
    // to avoid dealing with the special std::vector<bool> (i.e. bit array) interface.
    void populate_bseq_values(py::ptr<seq::Seq>, CompiledBasicSeq *cbseq,
                              std::vector<uint8_t> &chn_status);
    void eval_chn_actions(py::ptr<seq::Seq>, unsigned age);
    std::vector<ChannelAction*> &get_action_list(int chn, int bseq_id)
    {
        return all_chn_actions[chn + bseq_id * nchn];
    }
    ChannelAction *new_chn_action(int chn, int bseq_id)
    {
        auto res = chn_action_alloc.alloc();
        get_action_list(chn, bseq_id).push_back(res);
        return res;
    }
};

}

#endif
