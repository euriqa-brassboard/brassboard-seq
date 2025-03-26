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

#include <list>
#include <memory>
#include <vector>

namespace brassboard_seq::backend {

struct CompiledBasicSeq {
    int bseq_id;
    bool may_term;
    std::unique_ptr<std::vector<action::Action*>[]> all_actions;
    int64_t total_time;
    std::vector<int> next_bseq;
    std::vector<py_object> start_values;
};

struct CompiledSeq {
    std::vector<CompiledBasicSeq*> basic_seqs;
    std::list<CompiledBasicSeq> basic_seq_list;
};

}

#endif
