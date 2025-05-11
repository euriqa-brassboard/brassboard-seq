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

#include "action.h"
#include "config.h"
#include "event_time.h"
#include "scan.h"

#include <vector>

namespace brassboard_seq::seq {

using namespace rtval;
using event_time::EventTime;
using event_time::time_ref;

enum class TerminateStatus : uint8_t {
    Default,
    MayTerm,
    MayNotTerm,
};

struct CInfo {
    // Backtrace collection
    BacktraceTracker bt_tracker;
    action::ActionAllocator action_alloc;
    int action_counter{0};
};

struct SeqInfo : PyObject {
    // EventTime manager
    py::ref<event_time::TimeManager> time_mgr;
    // Global assumptions
    py::list_ref assertions;
    // Global config object
    py::ref<config::Config> config;
    // Name<->channel ID mapping
    py::dict_ref channel_name_map;
    py::dict_ref channel_path_map;
    py::list_ref channel_paths;
    py::ref<scan::ParamPack> C;
    std::shared_ptr<CInfo> cinfo;

    int get_channel_id(py::str name);
    py::str_ref channel_name_from_id(int cid)
    {
        return config::channel_name_from_path(channel_paths.get(cid));
    }

    static PyTypeObject Type;
};

struct TimeSeq : PyObject {
    // Toplevel parent sequence
    py::ref<SeqInfo> seqinfo;
    // The starting time of this sequence/step
    // This time point may or may not be related to the start time
    // of the parent sequence.
    time_ref start_time;
    // Ending time, for SubSeq this is also the time the next step is added by default
    time_ref end_time;
    // Condition for this sequence/step to be enabled.
    // This can be either a runtime value or `True` or `False`.
    // This is also always guaranteed to be true only if the parent's condition is true.
    py::ref<> cond;

    void show_cond_suffix(py::stringio &io) const;

    static PyTypeObject Type;
    constexpr static str_literal ClsName = "TimeSeq";
    using fields = field_pack<TimeSeq,&TimeSeq::seqinfo,&TimeSeq::start_time,
                              &TimeSeq::end_time,&TimeSeq::cond>;
};

struct TimeStep : TimeSeq {
    // This is the length that was passed in by the user (in unit of second)
    // to create the step without considering the condition if this step is enabled.
    // This is also the length parameter that'll be passed to the user function
    // if the action added to this step contains ramps.
    py::ref<> length;
    // The array of channel -> actions
    std::vector<action::Action*> actions;

    py::ptr<TimeStep> get_seq() { return this; }
    void show(py::stringio &io, int indent) const;
    template<bool is_pulse>
    void set(py::ptr<> chn, py::ptr<> value, py::ptr<> cond,
             bool exact_time, py::dict_ref &&kws);

    static PyTypeObject Type;
    constexpr static str_literal ClsName = "TimeStep";
    using fields = field_pack<TimeSeq::fields,&TimeStep::length>;
};

struct SubSeq : TimeSeq {
    // The list of subsequences and steps in this subsequcne
    py::list_ref sub_seqs;
    py::ref<TimeStep> dummy_step;

    py::ptr<SubSeq> get_seq() { return this; }
    void show_subseqs(py::stringio &io, int indent) const;
    void show(py::stringio &io, int indent) const;
    template<bool is_pulse>
    void set(py::ptr<> chn, py::ptr<> value, py::ptr<> cond,
             bool exact_time, py::dict_ref &&kws);
    void wait_cond(py::ptr<> length, py::ptr<> cond);
    void wait_for_cond(py::ptr<> _tp0, py::ptr<> offset, py::ptr<> cond);
    py::ref<SubSeq> add_custom_step(py::ptr<> cond, py::ptr<EventTime> start_time,
                                    py::ptr<> cb, size_t nargs, PyObject *const *args,
                                    py::tuple kwnames);
    py::ref<TimeStep> add_time_step(py::ptr<> cond, py::ptr<EventTime> start_time,
                                    py::ptr<> length);

    static PyTypeObject Type;
    constexpr static str_literal ClsName = "SubSeq";
    using fields = field_pack<TimeSeq::fields,&SubSeq::sub_seqs,&SubSeq::dummy_step>;
};

struct ConditionalWrapper : PyObject {
    py::ref<SubSeq> seq;
    py::ref<> cond;
    py::ptr<SubSeq> get_seq() { return seq; }
    void show(py::stringio &io, int indent) const;

    static PyTypeObject Type;
    constexpr static str_literal ClsName = "ConditionalWrapper";
};

struct BasicSeq : SubSeq {
    std::vector<int> next_bseq;
    int bseq_id;
    TerminateStatus term_status;
    py::list_ref basic_seqs;

    bool may_terminate() const
    {
        switch (term_status) {
        case TerminateStatus::MayTerm:
            return true;
        case TerminateStatus::MayNotTerm:
            return false;
        default:
            return next_bseq.empty();
        }
    }
    void add_branch(py::ptr<BasicSeq> bseq);
    void show_times(py::stringio &io, int indent) const;
    void show_next(py::stringio &io, int indent) const;
    void show(py::stringio &io, int indent) const;

    static PyTypeObject Type;
    constexpr static str_literal ClsName = "BasicSeq";
    using fields = field_pack<SubSeq::fields,&BasicSeq::basic_seqs>;
};

struct Seq : BasicSeq {
    bool inited{false};
    void show(py::stringio &io, int indent) const;

    static PyTypeObject Type;
    constexpr static str_literal ClsName = "Seq";
};

void init();

}

#endif
