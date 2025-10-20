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

#include "action.h"

namespace brassboard_seq::action {

BB_PROTECTED
void Action::show(py::stringio &io)
{
    io.write_ascii(is_pulse ? "Pulse(" : "Set(");
    rtval::show_value(io, value);
    if (cond != Py_True) {
        io.write_ascii(", cond=");
        rtval::show_value(io, cond);
    }
    if (exact_time)
        io.write_ascii(", exact_time=True");
    if (kws) {
        for (auto [name, val]: py::dict_iter(kws)) {
            io.write_ascii(", ");
            io.write(name);
            io.write_ascii("=");
            io.write_str(val);
        }
    }
    io.write_ascii(")");
}

BB_PROTECTED
PyTypeObject RampFunctionBase::Type = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "brassboard_seq.action.RampFunctionBase",
    .tp_basicsize = sizeof(RampFunctionBase),
    .tp_dealloc = py::tp_cxx_dealloc<true,RampFunctionBase>,
    .tp_flags = Py_TPFLAGS_DEFAULT,
};

__attribute__((visibility("internal")))
py::ref<> SeqCubicSpline::Data::eval_end(py::ptr<>, py::ptr<>)
{
    return order0 + order1 + order2 + order3;
}

__attribute__((visibility("internal")))
py::ref<> SeqCubicSpline::Data::spline_segments(double length, double oldval)
{
    f_inv_length = 1 / length;
    return py::new_tuple();
}

__attribute__((visibility("internal")))
bool SeqCubicSpline::Data::set_runtime_params(unsigned age)
{
    auto changed = tracked_assign(sp.order0, rtval::get_value_f64(order0, age));
    changed |= tracked_assign(sp.order1, rtval::get_value_f64(order1, age));
    changed |= tracked_assign(sp.order2, rtval::get_value_f64(order2, age));
    return changed | tracked_assign(sp.order3, rtval::get_value_f64(order3, age));
}

__attribute__((visibility("internal")))
rtval::TagVal SeqCubicSpline::Data::runtime_eval(double t) noexcept
{
    t = t * f_inv_length;
    return rtval::TagVal(sp.order0 + (sp.order1 + (sp.order2 + sp.order3 * t) * t) * t);
}

BB_PROTECTED
PyTypeObject SeqCubicSpline::Type = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "brassboard_seq.action.SeqCubicSpline",
    .tp_basicsize = sizeof(RampFunctionBase) + sizeof(SeqCubicSpline::Data),
    .tp_dealloc = py::tp_cxx_dealloc<true,SeqCubicSpline>,
    .tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_HAVE_GC,
    .tp_traverse = traverse<>,
    .tp_clear = clear<>,
    .tp_methods = (
        py::meth_table<
        py::meth_fast<"from_values",[] (auto, PyObject *const *args, Py_ssize_t nargs) {
            py::check_num_arg("SeqCubicSpline.from_values", nargs, 4, 4);
            py::ptr v0(args[0]);
            py::ptr v1(args[1]);
            py::ptr v2(args[2]);
            py::ptr v3(args[3]);

            auto f4_5 = py::new_float(4.5);
            auto fm5_5 = py::new_float(-5.5);

            // v0,
            // -5.5 * v0 + 9 * v1 - 4.5 * v2 + v3,
            // 9 * v0 - 22.5 * v1 + 18 * v2 - 4.5 * v3,
            // -4.5 * v0 + 13.5 * v1 - 13.5 * v2 + 4.5 * v3,
            return alloc(v0,
                         fm5_5 * v0 + py::int_cached(9) * v1 - f4_5 * v2 + v3,
                         f4_5 * (py::int_cached(2) * v0
                                 - py::int_cached(5) * v1
                                 + py::int_cached(4) * v2 - v3),
                         f4_5 * (py::int_cached(3) * (v1 - v2) - v0 + v3));
        },"",METH_STATIC>>),
    .tp_getset = (py::getset_table<
                  py::getset_def<"order0",[] (py::ptr<SeqCubicSpline> self) {
                      return py::newref(self->data()->order0); }>,
                  py::getset_def<"order1",[] (py::ptr<SeqCubicSpline> self) {
                      return py::newref(self->data()->order1); }>,
                  py::getset_def<"order2",[] (py::ptr<SeqCubicSpline> self) {
                      return py::newref(self->data()->order2); }>,
                  py::getset_def<"order3",[] (py::ptr<SeqCubicSpline> self) {
                      return py::newref(self->data()->order3); }>>),
    .tp_base = &RampFunctionBase::Type,
    .tp_vectorcall = py::vectorfunc<[] (auto, PyObject *const *args,
                                        ssize_t nargs, py::tuple kwnames) {
        py::check_num_arg("SeqCubicSpline.__init__", nargs, 0, 4);
        auto [order0, order1, order2, order3] =
            py::parse_pos_or_kw_args<"order0","order1","order2","order3">(
                "SeqCubicSpline.__init__", args, nargs, kwnames);
        py::check_required_pos_arg(order0, "SeqCubicSpline.__init__", "order0");
        if (!order1)
            order1 = py::int_cached(0);
        if (!order2)
            order2 = py::int_cached(0);
        if (!order3)
            order3 = py::int_cached(0);
        return alloc(order0, order1, order2, order3);
    }>
};

namespace {

struct RampFunction : RampFunctionBase::Base<RampFunction> {
    struct Data final : RampFunctionBase::Data {
        py::ref<> eval;
        py::ref<> _spline_segments;
        py::ref<> fvalue;
        std::unique_ptr<rtval::InterpFunction> interp_func;

        Data(py::ptr<PyTypeObject> type)
        {
            eval = type.attr("eval"_py);
            _spline_segments = type.try_attr("spline_segments"_py);
        }

        py::ptr<> py_self()
        {
            return py::ptr<>(((char*)this) - sizeof(RampFunctionBase));
        }

        py::ref<> call_eval(py::ptr<> t, py::ptr<> length, py::ptr<> oldval)
        {
            return eval(py_self(), t, length, oldval);
        }
        void compile()
        {
            static py::ptr arg0 = rtval::new_arg(py::int_cached(0), &PyFloat_Type).rel();
            static py::ptr arg1 = rtval::new_arg(py::int_cached(1), &PyFloat_Type).rel();
            static py::ptr arg2 = rtval::new_arg(py::int_cached(2), &PyFloat_Type).rel();
            static rtval::rtval_ptr const0 = rtval::new_const(0.0).rel();
            fvalue = call_eval(arg0, arg1, arg2);
            if (rtval::is_rtval(fvalue)) {
                interp_func.reset(new rtval::InterpFunction);
                std::vector<rtval::DataType> args{
                    rtval::DataType::Float64,
                    rtval::DataType::Float64,
                    rtval::DataType::Float64};
                if (rtval::rtval_ptr(fvalue)->datatype != rtval::DataType::Float64)
                    fvalue = rtval::new_expr2(rtval::Add, fvalue, const0);
                interp_func->set_value(fvalue, args);
            }
            else if (!fvalue.typeis<py::float_>()) {
                fvalue = to_py(fvalue.as_float());
            }
        }

        py::ref<> eval_end(py::ptr<> length, py::ptr<> oldval) override
        {
            return call_eval(length, length, oldval);
        }
        py::ref<> spline_segments(double length, double oldval) override
        {
            if (interp_func) {
                interp_func->data[1].f64_val = length;
                interp_func->data[2].f64_val = oldval;
            }
            if (!_spline_segments)
                return py::new_none();
            return _spline_segments(py_self(), to_py(length), to_py(oldval));
        }
        bool set_runtime_params(unsigned age) override
        {
            if (!fvalue)
                py_throw_format(PyExc_RuntimeError, "RampFunction.__init__ not called");
            if (interp_func)
                return interp_func->eval_all(age);
            return false;
        }
        rtval::TagVal runtime_eval(double t) noexcept override
        {
            if (!interp_func)
                return rtval::TagVal(PyFloat_AS_DOUBLE(fvalue.get()));
            interp_func->data[0].f64_val = t;
            return interp_func->call();
        }
    };

    using fields = field_pack<Data,&Data::eval,&Data::_spline_segments,&Data::fvalue>;
    static PyTypeObject Type;
};

PyTypeObject RampFunction::Type = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "brassboard_seq.action.RampFunction",
    .tp_basicsize = sizeof(RampFunctionBase) + sizeof(RampFunction::Data),
    .tp_dealloc = py::tp_cxx_dealloc<true,RampFunction>,
    .tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE|Py_TPFLAGS_HAVE_GC,
    .tp_traverse = traverse<>,
    .tp_clear = clear<>,
    .tp_base = &RampFunctionBase::Type,
    .tp_init = py::itrifunc<[] (py::ptr<RampFunction> self, py::tuple args, py::dict kws) {
        if (args)
            py::check_num_arg("RampFunction.__init__", args.size(), 0, 0);
        if (kws) {
            for (auto [name, value]: py::dict_iter(kws)) {
                self.set_attr(name, value);
            }
        }
        self->data()->compile();
    }>,
    .tp_new = py::tp_new<[] (PyTypeObject *t, auto...) {
        auto self = py::generic_alloc<RampFunction>(t);
        call_constructor(self->data(), t);
        return self;
    }>,
};

// These ramp functions can be implemented in python code but are provided here
// to be slightly more efficient.
struct Blackman : RampFunctionBase::Base<Blackman> {
    struct Data final : RampFunctionBase::Data {
        double f_amp;
        double f_offset;
        double f_t_scale;

        py::ref<> amp;
        py::ref<> offset;

        Data(py::ptr<> amp, py::ptr<> offset)
            : amp(amp.ref()), offset(offset.ref())
        {}

        py::ref<> eval_end(py::ptr<>, py::ptr<>) override
        {
            return offset.ref();
        }
        py::ref<> spline_segments(double length, double oldval) override
        {
            f_t_scale = length == 0 ? 0.0 : (M_PI * 2 / length);
            return py::new_none();
        }
        bool set_runtime_params(unsigned age) override
        {
            auto changed = tracked_assign(f_amp, rtval::get_value_f64(amp, age));
            return changed | tracked_assign(f_offset, rtval::get_value_f64(offset, age));
        }
        rtval::TagVal runtime_eval(double t) noexcept override
        {
            auto cost = cos(t * f_t_scale);
            auto val = f_amp * (0.34 - cost * (0.5 - 0.16 * cost));
            return rtval::TagVal(val + f_offset);
        }
    };

    using fields = field_pack<Data,&Data::amp,&Data::offset>;
    static PyTypeObject Type;
};

PyTypeObject Blackman::Type = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "brassboard_seq.action.Blackman",
    .tp_basicsize = sizeof(RampFunctionBase) + sizeof(Blackman::Data),
    .tp_dealloc = py::tp_cxx_dealloc<true,Blackman>,
    .tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_HAVE_GC,
    .tp_traverse = traverse<>,
    .tp_clear = clear<>,
    .tp_getset = (py::getset_table<
                  py::getset_def<"amp",[] (py::ptr<Blackman> self) {
                      return py::newref(self->data()->amp); }>,
                  py::getset_def<"offset",[] (py::ptr<Blackman> self) {
                      return py::newref(self->data()->offset); }>>),
    .tp_base = &RampFunctionBase::Type,
    .tp_vectorcall = py::vectorfunc<[] (auto, PyObject *const *args,
                                        ssize_t nargs, py::tuple kwnames) {
        py::check_num_arg("Blackman.__init__", nargs, 0, 2);
        auto [amp, offset] =
            py::parse_pos_or_kw_args<"amp","offset">(
                "Blackman.__init__", args, nargs, kwnames);
        py::check_required_pos_arg(amp, "Blackman.__init__", "amp");
        if (!offset)
            offset = py::int_cached(0);
        return alloc(amp, offset);
    }>
};

struct BlackmanSquare : RampFunctionBase::Base<BlackmanSquare> {
    struct Data final : RampFunctionBase::Data {
        double f_amp;
        double f_offset;
        double f_t_scale;

        py::ref<> amp;
        py::ref<> offset;

        Data(py::ptr<> amp, py::ptr<> offset)
            : amp(amp.ref()), offset(offset.ref())
        {}

        py::ref<> eval_end(py::ptr<>, py::ptr<>) override
        {
            return offset.ref();
        }
        py::ref<> spline_segments(double length, double oldval) override
        {
            f_t_scale = length == 0 ? 0.0 : (M_PI * 2 / length);
            return py::new_none();
        }
        bool set_runtime_params(unsigned age) override
        {
            auto changed = tracked_assign(f_amp, rtval::get_value_f64(amp, age));
            return changed | tracked_assign(f_offset, rtval::get_value_f64(offset, age));
        }
        rtval::TagVal runtime_eval(double t) noexcept override
        {
            auto cost = cos(t * f_t_scale);
            auto val = 0.34 - cost * (0.5 - 0.16 * cost);
            val = f_amp * val * val;
            return rtval::TagVal(val + f_offset);
        }
    };

    using fields = field_pack<Data,&Data::amp,&Data::offset>;
    static PyTypeObject Type;
};

PyTypeObject BlackmanSquare::Type = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "brassboard_seq.action.BlackmanSquare",
    .tp_basicsize = sizeof(RampFunctionBase) + sizeof(BlackmanSquare::Data),
    .tp_dealloc = py::tp_cxx_dealloc<true,BlackmanSquare>,
    .tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_HAVE_GC,
    .tp_traverse = traverse<>,
    .tp_clear = clear<>,
    .tp_getset = (py::getset_table<
                  py::getset_def<"amp",[] (py::ptr<BlackmanSquare> self) {
                      return py::newref(self->data()->amp); }>,
                  py::getset_def<"offset",[] (py::ptr<BlackmanSquare> self) {
                      return py::newref(self->data()->offset); }>>),
    .tp_base = &RampFunctionBase::Type,
    .tp_vectorcall = py::vectorfunc<[] (auto, PyObject *const *args,
                                        ssize_t nargs, py::tuple kwnames) {
        py::check_num_arg("BlackmanSquare.__init__", nargs, 0, 2);
        auto [amp, offset] =
            py::parse_pos_or_kw_args<"amp","offset">(
                "BlackmanSquare.__init__", args, nargs, kwnames);
        py::check_required_pos_arg(amp, "BlackmanSquare.__init__", "amp");
        if (!offset)
            offset = py::int_cached(0);
        return alloc(amp, offset);
    }>
};

struct LinearRamp : RampFunctionBase::Base<LinearRamp> {
    struct Data final : RampFunctionBase::Data {
        double f_start;
        double f_end;
        double f_inv_length;

        py::ref<> start;
        py::ref<> end;

        Data(py::ptr<> start, py::ptr<> end)
            : start(start.ref()), end(end.ref())
        {}

        py::ref<> eval_end(py::ptr<>, py::ptr<>) override
        {
            return end.ref();
        }
        py::ref<> spline_segments(double length, double oldval) override
        {
            f_inv_length = 1 / length;
            return py::new_tuple();
        }
        bool set_runtime_params(unsigned age) override
        {
            auto changed = tracked_assign(f_start, rtval::get_value_f64(start, age));
            return changed | tracked_assign(f_end, rtval::get_value_f64(end, age));
        }
        rtval::TagVal runtime_eval(double t) noexcept override
        {
            t = t * f_inv_length;
            return rtval::TagVal(f_start * (1 - t) + f_end * t);
        }
    };

    using fields = field_pack<Data,&Data::start,&Data::end>;
    static PyTypeObject Type;
};

PyTypeObject LinearRamp::Type = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "brassboard_seq.action.LinearRamp",
    .tp_basicsize = sizeof(RampFunctionBase) + sizeof(LinearRamp::Data),
    .tp_dealloc = py::tp_cxx_dealloc<true,LinearRamp>,
    .tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_HAVE_GC,
    .tp_traverse = traverse<>,
    .tp_clear = clear<>,
    .tp_getset = (py::getset_table<
                  py::getset_def<"start",[] (py::ptr<LinearRamp> self) {
                      return py::newref(self->data()->start); }>,
                  py::getset_def<"end",[] (py::ptr<LinearRamp> self) {
                      return py::newref(self->data()->end); }>>),
    .tp_base = &RampFunctionBase::Type,
    .tp_vectorcall = py::vectorfunc<[] (auto, PyObject *const *args,
                                        ssize_t nargs, py::tuple kwnames) {
        py::check_num_arg("LinearRamp.__init__", nargs, 0, 2);
        auto [start, end] =
            py::parse_pos_or_kw_args<"start","end">(
                "LinearRamp.__init__", args, nargs, kwnames);
        py::check_required_pos_arg(start, "LinearRamp.__init__", "start");
        py::check_required_pos_arg(end, "LinearRamp.__init__", "end");
        return alloc(start, end);
    }>
};

}

PyTypeObject &RampFunction_Type = RampFunction::Type;
PyTypeObject &Blackman_Type = Blackman::Type;
PyTypeObject &BlackmanSquare_Type = BlackmanSquare::Type;
PyTypeObject &LinearRamp_Type = LinearRamp::Type;

__attribute__((visibility("hidden")))
void init()
{
    throw_if(PyType_Ready(&RampFunctionBase::Type) < 0);
    throw_if(PyType_Ready(&RampFunction::Type) < 0);
    throw_if(PyType_Ready(&SeqCubicSpline::Type) < 0);
    throw_if(PyType_Ready(&Blackman::Type) < 0);
    throw_if(PyType_Ready(&BlackmanSquare::Type) < 0);
    throw_if(PyType_Ready(&LinearRamp::Type) < 0);
}

}
