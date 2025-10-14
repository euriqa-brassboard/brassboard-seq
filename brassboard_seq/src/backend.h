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

#include <memory>
#include <vector>

namespace brassboard_seq::backend {

using seq::BasicSeq;

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

struct SeqCompiler : PyObject {
    py::ref<seq::Seq> seq;
    int nchn;
    int nbseq;
    bool compiled;
    bool force_recompile;
    std::vector<bool> channel_changed;
    std::vector<CompiledBasicSeq> basic_cseqs;
    PermAllocator<ChannelAction,16> chn_action_alloc;
    std::vector<std::vector<ChannelAction*>> all_chn_actions;
    py::dict_ref backends;

    int visit_bseq(py::ptr<BasicSeq> bseq, std::vector<uint8_t> &visit_status);
    void initialize_bseqs();
    void initialize_actions();
    void populate_values();
    // Use std::vector<uint8_t> to pass in the status rather than std::vector<bool>
    // to avoid dealing with the special std::vector<bool> (i.e. bit array) interface.
    void populate_bseq_values(CompiledBasicSeq &cbseq, std::vector<uint8_t> &chn_status);
    void eval_chn_actions(unsigned age, bool isfirst);
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

    void finalize();
    void runtime_finalize(py::ptr<>);

    static PyTypeObject Type;
};

struct BackendBase : py::VBase<BackendBase> {
private:
    template<typename T> static constexpr auto _traverse =
        py::tp_traverse<[] (py::ptr<T> self, auto &visitor) {
            py::field_pack_visit<typename T::fields>(self->data(), visitor); }>;
    template<typename T> static constexpr auto _clear =
        py::iunifunc<[] (py::ptr<T> self) {
            py::field_pack_clear<typename T::fields>(self->data()); }>;
public:
    struct Data {
        py::str_ref prefix;
        virtual void finalize(py::ptr<SeqCompiler>) {}
        virtual void runtime_finalize(py::ptr<SeqCompiler>, unsigned, bool) {}
    };
    template<typename T>
    struct Base : py::VBase<BackendBase>::Base<T> {
    protected:
        template<typename=void> static constexpr auto traverse = _traverse<T>;
        template<typename=void> static constexpr auto clear = _clear<T>;
    };

    using fields = field_pack<Data>;
    static PyTypeObject Type;
};

extern PyTypeObject &Backend_Type;

void init();

}

#endif
