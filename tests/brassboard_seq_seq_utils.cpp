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

#include "test_utils.h"
#include "src/action.h"
#include "src/backend.h"
#include "src/event_time.h"
#include "src/seq.h"

using namespace brassboard_seq;

namespace {

struct Action : PyObject {
    std::unique_ptr<action::Action> tofree;
    action::Action *action;
    py::ref<> ref;

    static auto ref_action(action::Action *p, py::ptr<> parent)
    {
        auto self = py::generic_alloc<Action>();
        call_constructor(&self->ref, parent.ref());
        call_constructor(&self->tofree);
        self->action = p;
        return self;
    }

    static PyTypeObject Type;
};
PyTypeObject Action::Type = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "Action",
    .tp_basicsize = sizeof(Action),
    .tp_dealloc = py::tp_cxx_dealloc<true,Action>,
    .tp_str = py::unifunc<[] (py::ptr<Action> self) {
        py::stringio io;
        self->action->print(io);
        return io.getvalue();
    }>,
    .tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_HAVE_GC,
    .tp_traverse = py::tp_field_traverse<Action,&Action::ref>,
    .tp_clear = py::tp_field_clear<Action,&Action::ref>,
    .tp_methods = (
        py::meth_table<
        py::meth_o<"set_tid",[] (py::ptr<Action> self, py::ptr<> tid) {
            self->action->tid = tid.as_int();
        }>,
        py::meth_noargs<"get_aid",[] (py::ptr<Action> self) {
            return to_py(self->action->aid);
        }>,
        py::meth_noargs<"get_is_pulse",[] (py::ptr<Action> self) {
            return py::new_bool(self->action->is_pulse);
        }>,
        py::meth_noargs<"get_exact_time",[] (py::ptr<Action> self) {
            return py::new_bool(self->action->exact_time);
        }>,
        py::meth_noargs<"get_cond",[] (py::ptr<Action> self) {
            return py::ptr(self->action->cond).ref();
        }>,
        py::meth_noargs<"get_cond_val",[] (py::ptr<Action> self) {
            return py::new_bool(self->action->cond_val);
        }>,
        py::meth_noargs<"get_value",[] (py::ptr<Action> self) {
            return py::ptr(self->action->value).ref();
        }>,
        py::meth_noargs<"get_compile_info",[] (py::ptr<Action> self) {
            auto res = py::new_dict();
            res.set("tid", to_py(self->action->tid));
            res.set("end_tid", to_py(self->action->end_tid));
            res.set("length", (self->action->length ? self->action->length.ref() :
                               py::new_none()));
            res.set("end_val", (self->action->end_val ? self->action->end_val.ref() :
                                py::new_none()));
            res.set("cond_val", to_py(self->action->cond_val));
            return res;
        }>>),
    .tp_vectorcall = py::vectorfunc<[] (auto, PyObject *const *args,
                                        ssize_t nargs, py::tuple kwnames) {
        py::check_num_arg("Action", nargs, 6, 6);
        py::check_no_kwnames("Action", kwnames);
        py::dict_ref kws;
        if (args[4] != Py_None)
            kws.assign(args[4]);
        auto self = py::generic_alloc<Action>();
        call_constructor(&self->ref);
        call_constructor(&self->tofree,
                         new action::Action(args[0], args[1], py::ptr(args[2]).as_bool(),
                                            py::ptr(args[3]).as_bool(), std::move(kws),
                                            py::ptr(args[5]).as_int()));
        self->action = self->tofree.get();
        return self;
    }>
};

struct RampTest : PyObject {
    py::ref<action::RampFunctionBase> func;
    py::ref<> length;
    py::ref<> oldval;

    static PyTypeObject Type;
};
PyTypeObject RampTest::Type = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "RampTest",
    .tp_basicsize = sizeof(RampTest),
    .tp_dealloc = py::tp_cxx_dealloc<true,RampTest>,
    .tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_HAVE_GC,
    .tp_traverse = py::tp_field_traverse<RampTest,&RampTest::func,&RampTest::length,&RampTest::oldval>,
    .tp_clear = py::tp_field_clear<RampTest,&RampTest::func,&RampTest::length,&RampTest::oldval>,
    .tp_methods = (
        py::meth_table<
        py::meth_noargs<"eval_compile_end",[] (py::ptr<RampTest> self) {
            return self->func->eval_end(self->length, self->oldval);
        }>,
        py::meth_fast<"eval_runtime",[] (py::ptr<RampTest> self, PyObject *const *args,
                                         ssize_t nargs) {
            py::check_num_arg("eval_runtime", nargs, 2, 2);
            auto age = py::ptr(args[0]).as_int();
            py::ptr ts = args[1];
            self->func->set_runtime_params(age);
            self->func->spline_segments(rtval::get_value_f64(self->length, age),
                                        rtval::get_value_f64(self->oldval, age));
            auto res = py::new_list(0);
            for (auto t: ts.generic_iter()) {
                auto v = self->func->runtime_eval(t.as_float());
                throw_py_error(v.err);
                res.append(to_py(v.val.f64_val));
            }
            return res;
        }>>),
    .tp_vectorcall = py::vectorfunc<[] (auto, PyObject *const *args,
                                        ssize_t nargs, py::tuple kwnames) {
        py::check_num_arg("RampTest", nargs, 3, 3);
        py::check_no_kwnames("RampTest", kwnames);
        auto func = py::arg_cast<action::RampFunctionBase>(args[0], "func");
        auto self = py::generic_alloc<RampTest>();
        call_constructor(&self->func, func.ref());
        call_constructor(&self->length, py::newref(args[1]));
        call_constructor(&self->oldval, py::newref(args[2]));
        return self;
    }>
};

} // (anonymous)

static PyModuleDef test_module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "brassboard_seq_seq_utils",
    .m_size = -1,
    .m_methods = (
        py::meth_table<
        py::meth_o<"isramp",[] (auto, py::ptr<> v) {
            return py::new_bool(action::isramp(v));
        }>,
        py::meth_fast<"ramp_get_spline_segments",[] (auto, PyObject *const *args,
                                                     Py_ssize_t nargs) {
            py::check_num_arg("ramp_get_spline_segments", nargs, 3, 3);
            auto func = py::arg_cast<action::RampFunctionBase>(args[0], "func");
            return func->spline_segments(py::ptr(args[1]).as_float(),
                                         py::ptr(args[2]).as_float());
        }>,
        py::meth_o<"round_time",[] (auto, py::ptr<> v) -> py::ref<> {
            if (rtval::is_rtval(v)) {
                return event_time::round_time_rt(v);
            }
            else {
                return to_py(event_time::round_time_int(v));
            }
        }>,
        py::meth_noargs<"new_time_manager",[] (auto) {
            return event_time::TimeManager::alloc();
        }>,
        py::meth_fast<"time_manager_new_time",[] (auto, PyObject *const *args,
                                                  Py_ssize_t nargs) {
            py::check_num_arg("time_manager_new_time", nargs, 6, 6);
            auto self = py::arg_cast<event_time::TimeManager>(args[0], "time_manager");
            py::ptr prev = args[1];
            py::ptr offset = args[2];
            bool floating = py::ptr(args[3]).as_bool();
            if (rtval::is_rtval(offset))
                return self->new_rt(prev, offset, args[4], args[5]);
            return self->new_int(prev, offset.as_int<int64_t>(),
                                 floating, args[4], args[5]);
        }>,
        py::meth_fast<"time_manager_new_round_time",[] (auto, PyObject *const *args,
                                                        Py_ssize_t nargs) {
            py::check_num_arg("time_manager_new_round_time", nargs, 5, 5);
            auto self = py::arg_cast<event_time::TimeManager>(args[0], "time_manager");
            return self->new_round(args[1], args[2], args[3], args[4]);
        }>,
        py::meth_o<"time_manager_finalize",[] (auto, py::ptr<event_time::TimeManager> self) {
            return self->finalize();
        }>,
        py::meth_fast<"time_manager_compute_all_times",[] (auto, PyObject *const *args,
                                                           Py_ssize_t nargs) {
            py::check_num_arg("time_manager_compute_all_times", nargs, 2, 2);
            auto self = py::arg_cast<event_time::TimeManager>(args[0], "time_manager");
            auto max_time = self->compute_all_times(py::ptr(args[1]).as_int());
            return py::new_tuple(to_py(max_time), to_py(self->time_values));
        }>,
        py::meth_o<"time_manager_get_time_status",[] (auto, py::ptr<event_time::TimeManager> self) {
            return to_py(self->time_status);
        }>,
        py::meth_o<"time_manager_nchain",[] (auto, py::ptr<event_time::TimeManager> self) {
            py::ptr times = self->event_times;
            if (times.size() == 0)
                return to_py(0);
            return to_py(times.get<event_time::EventTime>(0)->chain_pos.size());
        }>,
        py::meth_fast<"event_time_set_base",[] (auto, PyObject *const *args,
                                                Py_ssize_t nargs) {
            py::check_num_arg("event_time_set_base", nargs, 3, 3);
            event_time::time_ptr self = args[0];
            if (auto rtoffset = py::cast<rtval::RuntimeValue>(args[2])) {
                self->set_base_rt(args[1], rtoffset);
            }
            else {
                self->set_base_int(args[1], py::ptr(args[2]).template as_int<int64_t>());
            }
        }>,
        py::meth_o<"event_time_id",[] (auto, py::ptr<event_time::EventTime> self) {
            return to_py(self->data.id);
        }>,
        py::meth_o<"event_time_get_static",[] (auto, py::ptr<event_time::EventTime> self) {
            return to_py(self->data.get_static());
        }>,
        py::meth_fast<"event_time_is_ordered",[] (auto, PyObject *const *args,
                                                  Py_ssize_t nargs) {
            py::check_num_arg("event_time_is_ordered", nargs, 2, 2);
            auto t1 = py::arg_cast<event_time::EventTime>(args[0], "t1");
            auto t2 = py::arg_cast<event_time::EventTime>(args[1], "t2");
            auto o1 = event_time::is_ordered(t1, t2);
            auto o2 = event_time::is_ordered(t2, t1);
            switch (o1) {
            case event_time::NoOrder:
                assert(o2 == event_time::NoOrder);
                return "NoOrder"_py.ref();
            case event_time::OrderEqual:
                assert(o2 == event_time::OrderEqual);
                return "OrderEqual"_py.ref();
            case event_time::OrderBefore:
                assert(o2 == event_time::OrderAfter);
                return "OrderBefore"_py.ref();
            case event_time::OrderAfter:
                assert(o2 == event_time::OrderBefore);
                return "OrderAfter"_py.ref();
            default:
                assert(false);
            }
        }>,
        py::meth_o<"seq_get_channel_paths",[] (auto, py::ptr<seq::Seq> s) {
            return s->seqinfo->channel_paths.ref();
        }>,
        py::meth_fast<"seq_get_event_time",[] (auto, PyObject *const *args,
                                               Py_ssize_t nargs) {
            py::check_num_arg("seq_get_event_time", nargs, 3, 3);
            py::ptr basic_seqs = py::arg_cast<seq::Seq>(args[0], "s")->basic_seqs;
            auto s = basic_seqs.get<seq::BasicSeq>(py::ptr(args[2]).as_int());
            return s->seqinfo->time_mgr->event_times.get(py::ptr(args[1]).as_int()).ref();
        }>,
        py::meth_o<"seq_get_cond",[] (auto, py::ptr<> condseq) {
            if (auto cond = py::exact_cast<seq::ConditionalWrapper>(condseq))
                return py::newref(cond->cond);
            return py::newref(py::arg_cast<seq::TimeSeq>(condseq, "s")->cond);
        }>,
        py::meth_o<"compiler_num_basic_seq",[] (auto, py::ptr<backend::SeqCompiler> comp) {
            return to_py(comp->basic_cseqs.size());
        }>,
        py::meth_fast<"compiler_get_all_start_values",[] (auto, PyObject *const *args,
                                                          Py_ssize_t nargs) {
            py::check_num_arg("compiler_get_all_start_values", nargs, 2, 2);
            auto comp = py::arg_cast<backend::SeqCompiler>(args[0], "comp");
            auto cbseq_id = py::ptr(args[1]).as_int();
            return py::new_nlist(comp->nchn, [&] (int chn) {
                return comp->basic_cseqs[cbseq_id].chn_actions[chn]->start_value.ref();
            });
        }>,
        py::meth_fast<"compiler_get_all_actions",[] (auto, PyObject *const *args,
                                                          Py_ssize_t nargs) {
            py::check_num_arg("compiler_get_all_actions", nargs, 2, 2);
            auto comp = py::arg_cast<backend::SeqCompiler>(args[0], "comp");
            auto cbseq_id = py::ptr(args[1]).as_int();
            return py::new_nlist(comp->nchn, [&] (int chn) {
                auto &actions = comp->basic_cseqs[cbseq_id].chn_actions[chn]->actions;
                return py::new_nlist(actions.size(), [&] (int i) {
                    return Action::ref_action(actions[i], comp->seq);
                });
            });
        }>,
        py::meth_fast<"compiler_get_bseq_id",[] (auto, PyObject *const *args,
                                                 Py_ssize_t nargs) {
            py::check_num_arg("compiler_get_bseq_id", nargs, 2, 2);
            auto comp = py::arg_cast<backend::SeqCompiler>(args[0], "comp");
            auto cbseq_id = py::ptr(args[1]).as_int();
            return to_py(comp->basic_cseqs[cbseq_id].bseq_id);
        }>,
        py::meth_fast<"compiler_get_all_times",[] (auto, PyObject *const *args,
                                                   Py_ssize_t nargs) {
            py::check_num_arg("compiler_get_all_times", nargs, 2, 2);
            auto comp = py::arg_cast<backend::SeqCompiler>(args[0], "comp");
            auto &cbseq = comp->basic_cseqs[py::ptr(args[1]).as_int()];
            py::ptr time_mgr =
                comp->seq->basic_seqs.get<seq::BasicSeq>(cbseq.bseq_id)->seqinfo->time_mgr;
            return py::new_tuple(to_py(cbseq.total_time), to_py(time_mgr->time_values));
        }>,
        py::meth_fast<"compiler_get_next_cbseq",[] (auto, PyObject *const *args,
                                                   Py_ssize_t nargs) {
            py::check_num_arg("compiler_get_next_cbseq", nargs, 2, 2);
            auto comp = py::arg_cast<backend::SeqCompiler>(args[0], "comp");
            auto &cbseq = comp->basic_cseqs[py::ptr(args[1]).as_int()];
            auto res = cbseq.next_bseq;
            if (cbseq.may_term)
                res.push_back(-1);
            return to_py(res);
        }>,
        py::meth_fast<"compiler_check_action_reuse",[] (auto, PyObject *const *args,
                                                        Py_ssize_t nargs) {
            py::check_num_arg("compiler_check_action_reuse", nargs, 4, 4);
            auto comp = py::arg_cast<backend::SeqCompiler>(args[0], "comp");
            auto cbseq_id1 = py::ptr(args[1]).as_int();
            auto cbseq_id2 = py::ptr(args[2]).as_int();
            auto chn = py::ptr(args[3]).as_int();
            return py::new_bool(comp->basic_cseqs[cbseq_id1].chn_actions[chn] ==
                                comp->basic_cseqs[cbseq_id2].chn_actions[chn]);
        }>>),
};

PY_MODINIT(brassboard_seq_seq_utils, test_module)
{
    m.add_type(&Action::Type);
    m.add_type(&RampTest::Type);
}
