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

#ifndef BRASSBOARD_SEQ_SRC_ACTION_H
#define BRASSBOARD_SEQ_SRC_ACTION_H

#include "utils.h"

#include "rtval.h"

namespace brassboard_seq::action {

struct Action {
    py::ref<> value;
    py::ref<> cond;
    py::ref<> kws;
    bool is_pulse;
    bool exact_time;
    bool cond_val;
    int aid;
    int tid;
    int end_tid;
    py::ptr<> length;
    py::ref<> end_val;

    Action(py::ptr<> value, py::ptr<> cond,
           bool is_pulse, bool exact_time, py::dict_ref &&kws, int aid)
        : value(value.ref()),
          cond(cond.ref()),
          kws(std::move(kws)),
          is_pulse(is_pulse),
          exact_time(exact_time),
          cond_val(false),
          aid(aid)
    {
    }

    py::str_ref py_str();
};

using ActionAllocator = PermAllocator<Action,146>;

struct RampFunctionBase : PyObject {
    struct Data {
        virtual py::ref<> eval_end(py::ptr<> length, py::ptr<> oldval) = 0;
        // Currently this function is also used to pass the runtime length and oldval
        // info to the ramp function to be used in subsequent runtime_eval calls.
        // This may be moved into a different function if/when we have a caller
        // that only need one of the effects of this function.
        // Also note that this API mutates the object and currently means
        // we cannot compute multiple ramps concurrently.
        virtual py::ref<> spline_segments(double length, double oldval) = 0;
        virtual void set_runtime_params(unsigned age) = 0;
        virtual rtval::TagVal runtime_eval(double t) noexcept = 0;
    };

    py::ref<> eval_end(py::ptr<> length, py::ptr<> oldval)
    {
        return data(this)->eval_end(length, oldval);
    }
    py::ref<> spline_segments(double length, double oldval)
    {
        return data(this)->spline_segments(length, oldval);
    }
    void set_runtime_params(unsigned age)
    {
        data(this)->set_runtime_params(age);
    }
    rtval::TagVal runtime_eval(double t) noexcept
    {
        return data(this)->runtime_eval(t);
    }

    static PyTypeObject Type;
protected:
    template<typename T>
    static typename T::Data *data(T *p)
    {
        return (typename T::Data*)(((char*)p) + sizeof(RampFunctionBase));
    }
    template<typename T, typename... Args>
    static py::ref<T> alloc(Args&&... args)
    {
        auto self = py::generic_alloc<T>();
        call_constructor(data(self.get()), std::forward<Args>(args)...);
        return self;
    }
    template<typename T>
    static constexpr auto traverse =
        py::tp_traverse<[] (py::ptr<T> self, auto &visitor) {
            py::field_pack_visit<typename T::fields>(data(self.get()), visitor); }>;

    template<typename T>
    static constexpr auto clear =
        py::iunifunc<[] (py::ptr<T> self) {
            py::field_pack_clear<typename T::fields>(data(self.get())); }>;
};

static inline bool isramp(py::ptr<> obj)
{
    return py::isinstance_nontrivial(obj, &RampFunctionBase::Type);
}

struct SeqCubicSpline : RampFunctionBase {
    struct Data final : RampFunctionBase::Data {
        cubic_spline sp;
        double f_inv_length;

        py::ref<> order0;
        py::ref<> order1;
        py::ref<> order2;
        py::ref<> order3;

        Data(py::ptr<> o0, py::ptr<> o1, py::ptr<> o2, py::ptr<> o3)
            : order0(o0.ref()), order1(o1.ref()), order2(o2.ref()), order3(o3.ref())
        {}

        py::ref<> eval_end(py::ptr<> length, py::ptr<> oldval) override;
        py::ref<> spline_segments(double length, double oldval) override;
        void set_runtime_params(unsigned age) override;
        rtval::TagVal runtime_eval(double t) noexcept override;
    };
    ~SeqCubicSpline();

    cubic_spline spline()
    {
        return data(this)->sp;
    }

    using fields = field_pack<Data,&Data::order0,&Data::order1,&Data::order2,&Data::order3>;
    static PyTypeObject Type;
};

extern PyTypeObject &RampFunction_Type;
extern PyTypeObject &Blackman_Type;
extern PyTypeObject &BlackmanSquare_Type;
extern PyTypeObject &LinearRamp_Type;

void init();

}

#endif
