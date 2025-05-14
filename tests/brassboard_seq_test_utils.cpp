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
#include "src/rtval.h"
#include "src/seq.h"
#include "src/yaml.h"

using namespace brassboard_seq;

namespace {

struct IntCollector : PyObject {
    PermAllocator<int,13> alloc;

    static PyTypeObject Type;
};
PyTypeObject IntCollector::Type = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "IntCollector",
    .tp_basicsize = sizeof(IntCollector),
    .tp_dealloc = py::tp_cxx_dealloc<false,IntCollector>,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_methods = (
        py::meth_table<
        py::meth_o<"add_int",[] (py::ptr<IntCollector> self, py::ptr<> v) {
            *self->alloc.alloc() = v.as_int();
        }>,
        py::meth_noargs<"sum",[] (py::ptr<IntCollector> self) {
            int s = 0;
            for (int &i: self->alloc)
                s += i;
            return to_py(s);
        }>>),
    .tp_vectorcall = py::vectorfunc<[] (auto, PyObject *const *args,
                                        ssize_t nargs, py::tuple kwnames) {
        py::check_num_arg("IntCollector", nargs, 0, 0);
        py::check_no_kwnames("IntCollector", kwnames);
        auto self = py::generic_alloc<IntCollector>();
        call_constructor(&self->alloc);
        return self;
    }>
};

struct TestCallback : rtval::ExternCallback {
    py::ref<> cb;

    static rtval::TagVal test_callback_extern(TestCallback *self)
    {
        return rtval::TagVal::from_py(self->cb());
    }

    static rtval::TagVal test_callback_extern_age(TestCallback *self, unsigned age)
    {
        return rtval::TagVal::from_py(self->cb(to_py(age)));
    }

    static inline py::ref<TestCallback> alloc(py::ptr<> cb, bool has_age)
    {
        auto self = py::generic_alloc<TestCallback>();
        self->fptr = (has_age ? (void*)test_callback_extern_age :
                      (void*)test_callback_extern);
        call_constructor(&self->cb, py::newref(cb));
        return self;
    }
    static PyTypeObject Type;
};
PyTypeObject TestCallback::Type = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "TestCallback",
    .tp_basicsize = sizeof(TestCallback),
    .tp_dealloc = py::tp_cxx_dealloc<true,TestCallback>,
    .tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_HAVE_GC,
    .tp_traverse = py::tp_field_traverse<TestCallback,&TestCallback::cb>,
    .tp_clear = py::tp_field_clear<TestCallback,&TestCallback::cb>,
    .tp_base = &ExternCallback::Type,
};

struct IOBuff : PyObject {
    py::stringio io;
    static PyTypeObject Type;
};
PyTypeObject IOBuff::Type = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "IOBuff",
    .tp_basicsize = sizeof(IOBuff),
    .tp_dealloc = py::tp_cxx_dealloc<false,IOBuff>,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_methods = (
        py::meth_table<
        py::meth_o<"write",[] (py::ptr<IOBuff> self, py::ptr<> s) {
            self->io.write(py::arg_cast<py::str>(s, "s"));
        }>,
        py::meth_o<"write_ascii",[] (py::ptr<IOBuff> self, py::ptr<> s) {
            self->io.write_ascii(py::arg_cast<py::bytes>(s, "s").data());
        }>,
        py::meth_fast<"write_rep_ascii",[] (py::ptr<IOBuff> self, PyObject *const *args,
                                            ssize_t nargs) {
            py::check_num_arg("write_rep_ascii", nargs, 2, 2);
            self->io.write_rep_ascii(py::ptr(args[0]).as_int(),
                                     py::arg_cast<py::bytes>(args[1], "s").data());
        }>,
        py::meth_noargs<"getvalue",[] (py::ptr<IOBuff> self) {
            return self->io.getvalue();
        }>>),
    .tp_vectorcall = py::vectorfunc<[] (auto, PyObject *const *args,
                                        ssize_t nargs, py::tuple kwnames) {
        py::check_num_arg("IOBuff", nargs, 0, 0);
        py::check_no_kwnames("IOBuff", kwnames);
        auto self = py::generic_alloc<IOBuff>();
        call_constructor(&self->io);
        return self;
    }>
};

template<std::integral ELT,unsigned N,str_literal name>
struct PyBits : PyObject {
    Bits<ELT,N> bits;

    static PyTypeObject Type;
    static PySequenceMethods as_sequence;
    static PyNumberMethods as_number;
    static py::ref<PyBits> alloc(Bits<ELT,N> bits)
    {
        auto self = py::generic_alloc<PyBits>();
        call_constructor(&self->bits, bits);
        return self;
    }
};
using Bits_i32x5 = PyBits<int32_t,5,"Bits_i32x5">;
using Bits_i64x4 = PyBits<int64_t,4,"Bits_i64x4">;
using Bits_u64x4 = PyBits<uint64_t,4,"Bits_u64x4">;
using Bits_i8x43 = PyBits<int8_t,43,"Bits_i8x43">;
static inline auto dispatch_bits(py::ptr<> v, auto &&cb)
{
    if (auto bits = py::cast<Bits_i32x5>(v)) {
        return cb(bits);
    }
    else if (auto bits = py::cast<Bits_i64x4>(v)) {
        return cb(bits);
    }
    else if (auto bits = py::cast<Bits_u64x4>(v)) {
        return cb(bits);
    }
    else if (auto bits = py::cast<Bits_i8x43>(v)) {
        return cb(bits);
    }
    else {
        py_throw_format(PyExc_TypeError, "Unknown input type");
    }
}
template<std::integral ELT,unsigned N,str_literal name>
PyNumberMethods PyBits<ELT,N,name>::as_number = {
    .nb_bool = py::iunifunc<[] (py::ptr<PyBits> self) {
        return (int)bool(self->bits);
    }>,
    .nb_invert = py::unifunc<[] (py::ptr<PyBits> self) {
        return alloc(~self->bits);
    }>,
    .nb_lshift = py::binfunc<[] (py::ptr<PyBits> self, py::ptr<> i) {
        return alloc(self->bits << (int)i.as_int());
    }>,
    .nb_rshift = py::binfunc<[] (py::ptr<PyBits> self, py::ptr<> i) {
        return alloc(self->bits >> (int)i.as_int());
    }>,
    .nb_and = py::binfunc<[] (py::ptr<> v1, py::ptr<> v2) {
        return dispatch_bits(v1, [&] (auto bits1) {
            return dispatch_bits(v2, [&] (auto bits2) -> py::ref<> {
                auto bits = bits1->bits & bits2->bits;
                if constexpr (sizeof(bits1->bits) >= sizeof(bits2->bits)) {
                    return bits1->alloc(bits);
                }
                else {
                    return bits2->alloc(bits);
                }
            });
        });
    }>,
    .nb_xor = py::binfunc<[] (py::ptr<> v1, py::ptr<> v2) {
        return dispatch_bits(v1, [&] (auto bits1) {
            return dispatch_bits(v2, [&] (auto bits2) -> py::ref<> {
                auto bits = bits1->bits ^ bits2->bits;
                if constexpr (sizeof(bits1->bits) >= sizeof(bits2->bits)) {
                    return bits1->alloc(bits);
                }
                else {
                    return bits2->alloc(bits);
                }
            });
        });
    }>,
    .nb_or = py::binfunc<[] (py::ptr<> v1, py::ptr<> v2) {
        return dispatch_bits(v1, [&] (auto bits1) {
            return dispatch_bits(v2, [&] (auto bits2) -> py::ref<> {
                auto bits = bits1->bits | bits2->bits;
                if constexpr (sizeof(bits1->bits) >= sizeof(bits2->bits)) {
                    return bits1->alloc(bits);
                }
                else {
                    return bits2->alloc(bits);
                }
            });
        });
    }>,
    .nb_inplace_and = py::binfunc<[] (py::ptr<PyBits> self, py::ptr<> v) {
        dispatch_bits(v, [&] (auto other) { self->bits &= other->bits; });
        return self.ref();
    }>,
    .nb_inplace_xor = py::binfunc<[] (py::ptr<PyBits> self, py::ptr<> v) {
        dispatch_bits(v, [&] (auto other) { self->bits ^= other->bits; });
        return self.ref();
    }>,
    .nb_inplace_or = py::binfunc<[] (py::ptr<PyBits> self, py::ptr<> v) {
        dispatch_bits(v, [&] (auto other) { self->bits |= other->bits; });
        return self.ref();
    }>,
    .nb_index = py::unifunc<[] (py::ptr<PyBits> self) {
        return self->bits.to_pylong();
    }>,
};
template<std::integral ELT,unsigned N,str_literal name>
PySequenceMethods PyBits<ELT,N,name>::as_sequence = {
    .sq_item = py::sq_item<[] (py::ptr<PyBits> self, Py_ssize_t i) {
        return to_py(self->bits[i]);
    }>,
    .sq_ass_item = py::sq_ass_item<[] (py::ptr<PyBits> self, Py_ssize_t i, py::ptr<> v) {
        self->bits[i] = v.as_int<ELT>();
    }>
};
template<std::integral ELT,unsigned N,str_literal name>
PyTypeObject PyBits<ELT,N,name>::Type = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = name,
    .tp_basicsize = sizeof(PyBits),
    .tp_dealloc = py::tp_cxx_dealloc<false,PyBits>,
    .tp_repr = py::unifunc<[] (py::ptr<PyBits> self) {
        py::stringio io;
        self->bits.print(io, true);
        return io.getvalue();
    }>,
    .tp_as_number = &as_number,
    .tp_as_sequence = &as_sequence,
    .tp_str = py::unifunc<[] (py::ptr<PyBits> self) {
        py::stringio io;
        self->bits.print(io, false);
        return io.getvalue();
    }>,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_richcompare = py::tp_richcompare<[] (py::ptr<PyBits> v1, py::ptr<> _v2, int op) {
        auto v2 = py::arg_cast<PyBits>(_v2, "v2");
        switch (op) {
        default:
        case Py_EQ:
            return to_py(v1->bits == v2->bits);
        case Py_NE:
            return to_py(v1->bits != v2->bits);
        case Py_GT:
            return to_py(v1->bits > v2->bits);
        case Py_LT:
            return to_py(v1->bits < v2->bits);
        case Py_GE:
            return to_py(v1->bits >= v2->bits);
        case Py_LE:
            return to_py(v1->bits <= v2->bits);
        }
    }>,
    .tp_methods = (
        py::meth_table<
        py::meth_noargs<"bytes",[] (py::ptr<PyBits> self) {
            return self->bits.to_pybytes();
        }>,
        py::meth_noargs<"spec",[] (auto) {
            return py::new_tuple(to_py(N), to_py(sizeof(ELT) * 8),
                                 to_py(std::is_signed_v<ELT>));
        },"",METH_STATIC>,
        py::meth_fast<"get_mask",[] (auto, PyObject *const *args, Py_ssize_t nargs) {
            py::check_num_arg("get_mask", nargs, 2, 2);
            return alloc(Bits<ELT,N>::mask(py::ptr(args[0]).as_int(),
                                           py::ptr(args[1]).as_int()));
        },"",METH_STATIC>>),
    .tp_vectorcall = py::vectorfunc<[] (auto, PyObject *const *args,
                                        ssize_t nargs, py::tuple kwnames) {
        py::check_num_arg(name, nargs, 0, 1);
        py::check_no_kwnames(name, kwnames);
        auto self = py::generic_alloc<PyBits>();
        if (nargs == 0) {
            call_constructor(&self->bits, ELT(0));
        }
        else if (auto i = py::cast<py::int_>(args[0])) {
            call_constructor(&self->bits, i.as_int<ELT>());
        }
        else {
            dispatch_bits(args[0], [&] (auto bits) {
                call_constructor(&self->bits, bits->bits);
            });
        }
        return self;
    }>
};

} // (anonymous)

static auto to_vector_int(py::ptr<> ary)
{
    std::vector<int> V;
    for (auto v: ary.generic_iter())
        V.push_back(v.as_int());
    return V;
}

static PyModuleDef test_module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "brassboard_seq_test_utils",
    .m_size = -1,
    .m_methods = (
        py::meth_table<
        py::meth_fast<"cxx_error",[] (auto, PyObject *const *args,
                                      Py_ssize_t nargs) {
            py::check_num_arg("cxx_error", nargs, 2, 2);
            int type = py::ptr(args[0]).as_int();
            auto str = py::str(args[1]).utf8();
            switch (type) {
            case 0:
                throw std::bad_alloc();
            case 1:
                throw std::bad_cast();
            case 2:
                throw std::bad_typeid();
            case 3:
                throw std::domain_error(str);
            case 4:
                throw std::invalid_argument(str);
            case 5:
                throw std::ios_base::failure(str);
            case 6:
                throw std::out_of_range(str);
            case 7:
                throw std::overflow_error(str);
            case 8:
                throw std::range_error(str);
            case 9:
                throw std::underflow_error(str);
            case 10:
                throw std::exception();
            default:
                throw 0;
            }
        }>,
        py::meth_o<"get_suffix_array",[] (auto, py::ptr<> ary) {
            auto S = to_vector_int(ary);
            int N = S.size();
            std::vector<int> SA(N);
            std::vector<int> ws(N);
            get_suffix_array(SA, S, ws);
            return to_py(SA);
        }>,
        py::meth_fast<"get_height_array",[] (auto, PyObject *const *args,
                                             Py_ssize_t nargs) {
            py::check_num_arg("get_height_array", nargs, 2, 2);
            auto S = to_vector_int(args[0]);
            auto SA = to_vector_int(args[1]);
            int N = S.size();
            std::vector<int> RK(N);
            std::vector<int> height(N <= 2 ? 0 : N - 2);
            order_to_rank(RK, SA);
            get_height_array(height, S, SA, RK);
            return to_py(height);
        }>,
        py::meth_o<"get_max_range",[] (auto, py::ptr<> v) {
            auto value = to_vector_int(v);
            auto res = py::new_list(0);
            foreach_max_range(value, [&] (int i0, int i1, int maxv) {
                bool found_equal = false;
                for (int i = i0; i <= i1; i++) {
                    auto v = value[i];
                    assert(v >= maxv);
                    found_equal |= v == maxv;
                }
                assert(found_equal);
                res.append(py::new_tuple(to_py(i0), to_py(i1), to_py(maxv)));
            });
            return res;
        }>,
        py::meth_o<"int_to_chars",[] (auto, py::ptr<> v) {
            char buff[5];
            auto ptr = to_chars(buff, v.as_int());
            return py::new_bytes(buff, ptr - buff);
        }>,
        py::meth_o<"int_throw_if",[] (auto, py::ptr<> v) {
            return to_py(throw_if(v.as_int()));
        }>,
        py::meth_o<"int_throw_if_not",[] (auto, py::ptr<> v) {
            return to_py(throw_if_not(v.as_int()));
        }>,
        py::meth_fast<"check_num_arg",[] (auto, PyObject *const *args,
                                          Py_ssize_t nargs) {
            py::check_num_arg("check_num_arg", nargs, 4, 4);
            py::check_num_arg(py::arg_cast<py::bytes>(args[0], "func_name").data(),
                              py::ptr(args[1]).as_int(), py::ptr(args[2]).as_int(),
                              py::ptr(args[3]).as_int());
        }>,
        py::meth_fast<"yaml_io_print",[] (auto, PyObject *const *args, Py_ssize_t nargs) {
            py::check_num_arg("yaml_io_sprint", nargs, 1, 2);
            int indent = 0;
            if (nargs >= 2) {
                indent = py::arg_cast<py::int_>(args[1], "indent").as_int();
                if (indent < 0) {
                    py_throw_format(PyExc_TypeError, "indent cannot be negative");
                }
            }
            py::stringio io;
            yaml::print(io, args[0], indent);
            return io.getvalue();
        }>,
        py::meth_noargs<"new_invalid_rtval",[] (auto) {
            // This should only happen if something really wrong happens.
            // We'll just test that we behave reasonably enough.
            // (it's unavoidable that we'll crash in some cases)
            using namespace rtval;
            auto rt = new_expr2(Add, new_const(1), new_const(1));
            rt->type_ = (ValueType)1000;
            return rt;
        }>,
        py::meth_o<"new_const",[] (auto, py::ptr<> c) {
            return rtval::new_const(c);
        }>,
        py::meth_o<"new_arg",[] (auto, py::ptr<> idx) {
            return rtval::new_arg(idx, &PyFloat_Type);
        }>,
        py::meth_fast<"new_extern",[] (auto, PyObject *const *args,
                                       Py_ssize_t nargs) {
            py::check_num_arg("new_extern", nargs, 1, 2);
            py::ptr ty = (nargs > 1) ? args[1] : (PyObject*)&PyFloat_Type;
            return rtval::new_extern(TestCallback::alloc(args[0], false), ty);
        }>,
        py::meth_fast<"new_extern_age",[] (auto, PyObject *const *args,
                                           Py_ssize_t nargs) {
            py::check_num_arg("new_extern_age", nargs, 1, 2);
            py::ptr ty = (nargs > 1) ? args[1] : (PyObject*)&PyFloat_Type;
            return rtval::new_extern_age(TestCallback::alloc(args[0], true), ty);
        }>>),
};

PY_MODINIT(brassboard_seq_test_utils, test_module)
{
    m.add_type(&IntCollector::Type);
    m.add_type(&TestCallback::Type);
    m.add_type(&IOBuff::Type);
    m.add_type(&Bits_i32x5::Type);
    m.add_type(&Bits_i64x4::Type);
    m.add_type(&Bits_u64x4::Type);
    m.add_type(&Bits_i8x43::Type);
}
