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

#ifndef BRASSBOARD_SEQ_SRC_SEQ_H
#define BRASSBOARD_SEQ_SRC_SEQ_H

#include "utils.h"

namespace brassboard_seq::seq {

enum class TerminateStatus : uint8_t {
    Default,
    MayTerm,
    MayNotTerm,
};

static py_object channel_name_from_id(auto *seqinfo, int cid)
{
    return channel_name_from_path(PyList_GET_ITEM(seqinfo->channel_paths, cid));
}

static inline bool basicseq_may_terminate(auto self)
{
    auto term_status = pyx_fld(self, term_status);
    switch (term_status) {
    case TerminateStatus::MayTerm:
        return true;
    case TerminateStatus::MayNotTerm:
        return false;
    default:
        return pyx_fld(self, next_bseq).empty();
    }
}

}

#endif
