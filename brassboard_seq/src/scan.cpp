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

#include "scan.h"

#include "yaml.h"

#include "numpy/arrayobject.h"

namespace brassboard_seq::scan {

namespace {

// Check if the struct field reference path is overwritten in `obj`.
// Overwrite happens if the field itself exists or a parent of the field
// is overwritten to something that's not scalar struct.
static inline bool check_field(py::dict d, py::tuple path)
{
    for (auto [_, name]: py::tuple_iter(path)) {
        auto vp = d.try_get(name);
        if (!vp)
            return false;
        if (!vp.typeis<py::dict>())
            return true;
        d = vp;
    }
    return true;
}

static py::ptr<> recursive_get(py::dict d, py::tuple path)
{
    auto pathlen = path.size();
    if (!pathlen)
        return d;
    py::ptr<> vp;
    for (auto [i, f]: py::tuple_iter(path)) {
        vp = d.try_get(f);
        if (!vp)
            return py::ptr<>();
        if (i == pathlen - 1)
            break;
        if (!vp.typeis<py::dict>())
            return py::ptr<>();
        d = vp;
    }
    return vp;
}

static void recursive_assign(py::dict d, py::tuple path, py::ptr<> v)
{
    auto pathlen = path.size();
    assert(pathlen > 0);
    for (auto [i, f]: py::tuple_iter(path)) {
        if (i == pathlen - 1) {
            d.set(f, v);
            return;
        }
        auto vp = d.try_get(f);
        if (vp && vp.typeis<py::dict>()) {
            d = vp;
        }
        else {
            auto newd = py::new_dict();
            d.set(f, newd);
            d = newd; // the parent dict would be keeping newd alive.
        }
    }
}

static void _foreach_nondict(auto &cb, py::dict obj)
{
    for (auto [k, v]: py::dict_iter(obj)) {
        if (auto d = py::cast<py::dict>(v)) {
            _foreach_nondict(cb, d);
        }
        else {
            cb(v);
        }
    }
}

static void _foreach_nondict(auto &cb, py::dict obj, py::tuple prefix)
{
    for (auto [k, v]: py::dict_iter(obj)) {
        auto path = prefix.append(k);
        if (auto d = py::cast<py::dict>(v)) {
            _foreach_nondict(cb, d, path);
        }
        else {
            cb(v, path);
        }
    }
}

static void foreach_nondict(py::dict obj, auto &&cb)
{
    if constexpr (requires { cb(py::ptr<>(), py::tuple()); }) {
        _foreach_nondict(cb, obj, py::empty_tuple);
    }
    else {
        _foreach_nondict(cb, obj);
    }
}

template<typename T>
static inline auto load_cast(py::ptr<> obj, const char *name)
{
    if (auto res = py::exact_cast<T>(obj))
        return res;
    py_throw_format(PyExc_TypeError, "Invalid serialization of ScanGroup: "
                    "wrong %s type %S.", name, obj.type());
}

struct Scan1D : PyObject {
    int size;
    py::dict_ref params;

    py::ref<Scan1D> copy() const
    {
        auto self = py::generic_alloc<Scan1D>();
        self->size = size;
        call_constructor(&self->params, py::dict_deepcopy(params));
        return self;
    }
    py::dict_ref dump()
    {
        auto res = py::new_dict();
        res.set("size"_py, py::new_int(size));
        res.set("params"_py, params);
        return res;
    }

    static py::ref<Scan1D> alloc()
    {
        auto self = py::generic_alloc<Scan1D>();
        self->size = 0;
        call_constructor(&self->params, py::new_dict());
        return self;
    }
    static py::ref<Scan1D> load(py::dict data)
    {
        auto self = py::generic_alloc<Scan1D>();
        if (auto size = data.try_get("size"_py))
            self->size = size.as_int();
        if (auto params = data.try_get("params"_py)) {
            call_constructor(&self->params,
                             py::dict_deepcopy(load_cast<py::dict>(params, "parameter")));
            foreach_nondict(self->params, [&] (py::ptr<> v) {
                auto sz = load_cast<py::list>(v, "parameter").size();
                if (sz != self->size) {
                    py_throw_format(PyExc_ValueError, "Invalid serialization"
                                    " of ScanGroup: scan size mismatch, "
                                    "expect %d, got %d.", self->size, (int)sz);
                }
            });
        }
        else {
            call_constructor(&self->params, py::new_dict());
        }
        return self;
    }
    static PyTypeObject Type;
};
PyTypeObject Scan1D::Type = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "brassboard_seq.scan.Scan1D",
    .tp_basicsize = sizeof(Scan1D),
    .tp_dealloc = py::tp_cxx_dealloc<true,Scan1D>,
    .tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_HAVE_GC,
    .tp_traverse = py::tp_field_traverse<Scan1D,&Scan1D::params>,
    .tp_clear = py::tp_field_clear<Scan1D,&Scan1D::params>,
};

struct ScanND : PyObject {
    int baseidx;
    py::dict_ref fixed;
    py::list_ref vars;

    bool is_default()
    {
        return baseidx == -1 && fixed.size() == 0 && vars.size() == 0;
    }
    bool contains_path(py::tuple path)
    {
        if (check_field(fixed, path))
            return true;
        for (auto [i, var]: py::list_iter<Scan1D>(vars)) {
            if (check_field(var->params, path)) {
                return true;
            }
        }
        return false;
    }

    py::ref<ScanND> copy()
    {
        auto self = py::generic_alloc<ScanND>();
        self->baseidx = baseidx;
        call_constructor(&self->fixed, py::dict_deepcopy(fixed));
        call_constructor(&self->vars, py::new_nlist(vars.size(), [&] (int i) {
            return vars.get<Scan1D>(i)->copy();
        }));
        return self;
    }
    py::dict_ref dump(bool dumpbase)
    {
        auto res = py::new_dict();
        if (dumpbase)
            res.set("baseidx"_py, py::new_int(baseidx + 1));
        res.set("params"_py, fixed);
        res.set("vars"_py, py::new_nlist(vars.size(), [&] (int i) {
            return vars.get<Scan1D>(i)->dump();
        }));
        return res;
    }
    int size()
    {
        int res = 1;
        for (auto [i, var]: py::list_iter<Scan1D>(vars))
            res *= var->size == 0 ? 1 : var->size;
        return res;
    }
    py::dict_ref getseq(int seqidx)
    {
        int orig_idx = seqidx;
        auto seq = py::dict_deepcopy(fixed);
        for (auto [i, var]: py::list_iter<Scan1D>(vars)) {
            auto varsz = var->size;
            if (varsz == 0)
                continue;
            auto subidx = seqidx % varsz;
            seqidx = (seqidx - subidx) / varsz;
            foreach_nondict(var->params, [&] (py::list v, py::tuple path) {
                recursive_assign(seq, path, v.get(subidx));
            });
        }
        if (seqidx != 0)
            py_throw_format(PyExc_IndexError, "Sequence index out of bound: %d.",
                            orig_idx);
        return seq;
    }
    void show(py::stringio &io, int indent, py::tuple path)
    {
        bool empty = true;
        if (baseidx != -1) {
            empty = false;
            io.write_rep_ascii(indent, " ");
            io.write_ascii("Base index: ");
            io.write_cxx<32>(baseidx);
            io.write_ascii("\n");
        }
        auto new_indent = indent + 3;
        if (auto fixed = recursive_get(this->fixed, path);
            fixed && (!fixed.typeis<py::dict>() || py::dict(fixed).size() != 0)) {
            empty = false;
            io.write_rep_ascii(indent, " ");
            io.write_ascii("Fixed parameters:\n");
            io.write_rep_ascii(new_indent, " ");
            yaml::print(io, fixed, new_indent);
            io.write_ascii("\n");
        }
        for (auto [i, var]: py::list_iter<Scan1D>(vars)) {
            if (var->size == 0)
                continue;
            empty = false;
            io.write_rep_ascii(indent, " ");
            io.write_ascii("Scan dimension ");
            io.write_cxx<32>(i);
            io.write_ascii(": (size ");
            io.write_cxx<32>(var->size);
            io.write_ascii(")\n");
            io.write_rep_ascii(new_indent, " ");
            if (auto params = recursive_get(var->params, path)) {
                yaml::print(io, params, new_indent);
                io.write_ascii("\n");
            }
            else {
                io.write_ascii("<empty>\n");
            }
        }
        if (empty) {
            io.write_rep_ascii(indent, " ");
            io.write_ascii("<empty>\n");
        }
    }
    void check_noconflict(py::tuple path, int scandim)
    {
        if (scandim != -1 && check_field(fixed, path))
            py_throw_format(PyExc_ValueError, "Cannot scan a fixed parameter.");
        for (auto [i, var]: py::list_iter<Scan1D>(vars)) {
            if (i == scandim)
                continue;
            if (check_field(var->params, path)) {
                if (scandim == -1) {
                    py_throw_format(PyExc_ValueError, "Cannot fix a scanned parameter.");
                }
                else {
                    py_throw_format(PyExc_ValueError,
                                    "Cannot scan a parameter in multiple dimensions.");
                }
            }
        }
    }

    static py::ref<ScanND> alloc()
    {
        auto self = py::generic_alloc<ScanND>();
        self->baseidx = -1;
        call_constructor(&self->fixed, py::new_dict());
        call_constructor(&self->vars, py::new_list(0));
        return self;
    }
    static py::ref<ScanND> load(py::dict data)
    {
        auto self = py::generic_alloc<ScanND>();
        if (auto baseidx = data.try_get("baseidx"_py)) {
            self->baseidx = baseidx.as_int() - 1;
        }
        else {
            self->baseidx = -1;
        }
        if (auto fixed = data.try_get("params"_py)) {
            call_constructor(&self->fixed,
                             py::dict_deepcopy(load_cast<py::dict>(fixed, "parameter")));
        }
        else {
            call_constructor(&self->fixed, py::new_dict());
        }
        if (auto vars = data.try_get("vars"_py)) {
            auto var_list = load_cast<py::list>(vars, "variables");
            call_constructor(&self->vars, py::new_nlist(var_list.size(), [&] (int i) {
                return Scan1D::load(var_list.get(i));
            }));
        }
        else {
            call_constructor(&self->vars, py::new_list(0));
        }
        return self;
    }
    static PyTypeObject Type;
};
PyTypeObject ScanND::Type = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "brassboard_seq.scan.ScanND",
    .tp_basicsize = sizeof(ScanND),
    .tp_dealloc = py::tp_cxx_dealloc<true,ScanND>,
    .tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_HAVE_GC,
    .tp_traverse = py::tp_field_traverse<ScanND,&ScanND::fixed,&ScanND::vars>,
    .tp_clear = py::tp_field_clear<ScanND,&ScanND::fixed,&ScanND::vars>,
};

// Check if the assigning `obj` with subfield reference `path`
// will cause forbidden overwrite. This happens when `path` exists in `obj`
// and if either the original value of the new value is a scalar struct.
// These assignments are forbidding since we do not want any field to
// change type (struct -> non-struct or non-struct -> struct).
// We also don't allow assigning struct to struct even if there's no conflict
// in the structure of the old and new values since the old value might have
// individual fields explicitly assigned previously and this has caused a few
// surprises in practice...
// We could in principle change the semantics to treat the assigned value as default
// value (so it will not overwrite any existing fields)
// but that is not super consistent with other semantics.
// If the more restricted semantics (error on struct assignment to struct)
// is proven insufficient, we can always add the merge semantics later without breaking.
static void check_param_overwrite(py::dict obj, py::tuple path, bool isdict)
{
    auto pathlen = path.size();
    py::ptr<> vp;
    for (auto [i, p]: py::tuple_iter(path)) {
        vp = obj.try_get(p);
        if (!vp)
            return;
        if (i == pathlen - 1)
            break;
        obj = py::exact_cast<py::dict>(vp);
        if (!obj) {
            py_throw_format(PyExc_TypeError, "Assignment to field of scalar not allowed.");
        }
    }
    auto wasdict = vp.typeis<py::dict>();
    if (isdict && !wasdict) {
        py_throw_format(PyExc_TypeError,
                        "Changing field from non-dict to dict not allowed.");
    }
    else if (!isdict && wasdict) {
        py_throw_format(PyExc_TypeError,
                        "Changing field from dict to non-dict not allowed.");
    }
    else if (isdict && wasdict) {
        // See comment above for explaination.
        py_throw_format(PyExc_TypeError, "Override dict not allowed.");
    }
}

static py::ref<> convert_param_value(py::ptr<> obj, bool allow_dict)
{
    if (PyArray_IsScalar(obj, Bool))
        return py::new_bool(obj.as_bool());
    if (PyArray_IsScalar(obj, Integer))
        return py::new_int(obj.as_int());
    if (PyArray_IsScalar(obj, Floating))
        return py::new_float(obj.as_float());
    if (auto list = py::cast<py::list>(obj))
        return py::new_nlist(list.size(), [&] (int i) {
            return convert_param_value(list.get(i), true);
        });
    if (PyArray_Check(obj)) {
        auto ary = (PyArrayObject*)obj;
        if (PyArray_NDIM(ary) != 1)
            py_throw_format(PyExc_TypeError, "Scan only support 1D array");
        auto data = (char*)PyArray_DATA(ary);
        auto sz = PyArray_DIM(ary, 0);
        auto elsz = PyArray_ITEMSIZE(ary);
        return py::new_nlist(sz, [&] (int i) {
            auto item = py::ref<>::checked(PyArray_GETITEM(ary, data + i * elsz));
            return convert_param_value(item, true);
        });
    }
    if (auto dict = py::cast<py::dict>(obj)) {
        if (!allow_dict)
            py_throw_format(PyExc_TypeError, "Scan parameter cannot be a dict.");
        auto res = py::new_dict();
        for (auto [k, v]: py::dict_iter(dict))
            res.set(k, convert_param_value(v, true));
        return res;
    }
    return obj.ref();
}

struct ScanGroup : PyObject {
    py::ref<ScanND> base;
    py::list_ref scans;
    py::list_ref scanscache;
    bool new_empty_called;

    void add_empty(int n=1)
    {
        for (int i = 0; i < n; i++) {
            scans.append(ScanND::alloc());
            scanscache.append(Py_None);
        }
    }
    void set_dirty(int idx)
    {
        assert(idx < scanscache.size());
        if (idx < 0) {
            for (int i = 0, n = scanscache.size(); i < n; i++) {
                scanscache.set(i, py::new_none());
            }
        }
        else {
            scanscache.set(idx, py::new_none());
        }
    }
    py::ptr<ScanND> getfullscan(int idx, bool allow_base)
    {
        if (idx == -1 && allow_base)
            return base.ptr();
        int nscans = scanscache.size();
        if (idx < 0 || idx >= nscans)
            py_throw_format(PyExc_IndexError, "Scan group index out of bound: %d.", idx);
        if (auto cache = scanscache.get<ScanND>(idx); !cache.is_none())
            return cache;
        auto scan = scans.get<ScanND>(idx)->copy();
        auto base = getfullscan(scan->baseidx, true);
        // Merge the fixed parameters
        foreach_nondict(base->fixed, [&] (py::ptr<> v, py::tuple path) {
            if (scan->contains_path(path))
                return;
            recursive_assign(scan->fixed, path, v);
        });
        // Merge the variable parameters
        for (auto [scanid, var]: py::list_iter<Scan1D>(base->vars)) {
            foreach_nondict(var->params, [&] (py::ptr<> v, py::tuple path) {
                if (scan->contains_path(path))
                    return;
                py::list vars = scan->vars;
                for (int i = 0, n = scanid - vars.size() + 1; i < n; i++)
                    vars.append(Scan1D::alloc());
                recursive_assign(vars.get<Scan1D>(scanid)->params, path, v);
            });
        }
        for (auto [scanid, s1d]: py::list_iter<Scan1D>(scan->vars)) {
            int sz = 0;
            foreach_nondict(s1d->params, [&] (py::list v) {
                int nv = v.size();
                assert(nv > 1);
                if (sz == 0) {
                    sz = nv;
                }
                else if (sz != nv) {
                    py_throw_format(PyExc_ValueError,
                                    "Scan parameter size does not match.");
                }
            });
            s1d->size = sz;
        }
        py::ptr scanp = scan;
        scanscache.set(idx, std::move(scan));
        return scanp;
    }
    void add_cat_scannd(py::ptr<ScanGroup> other, int idx)
    {
        auto scan = other->getfullscan(idx, true)->copy();
        scan->baseidx = -1;
        scans.append(std::move(scan));
        scanscache.append(py::new_none());
    }
    void add_cat_scan(py::ptr<> scan);
    void set_fixed_param(py::ptr<ScanND> scan, py::tuple path, int idx, py::ptr<> v);
    void set_scan_param(py::ptr<ScanND> scan, py::tuple path, int idx,
                        int scandim, py::ptr<> v);

    static py::ref<ScanGroup> load_v1(py::dict data)
    {
        auto self = py::generic_alloc<ScanGroup>();
        // self->new_empty_called = false;
        if (auto base = data.try_get("base"_py)) {
            call_constructor(&self->base, ScanND::load(base));
        }
        else {
            call_constructor(&self->base, ScanND::alloc());
        }
        int nscans = 1;
        if (auto _scans = data.try_get("scans"_py)) {
            auto scans = load_cast<py::list>(_scans, "scans");
            nscans = scans.size();
            call_constructor(&self->scans, py::new_nlist(nscans, [&] (auto i) {
                return ScanND::load(scans.get(i));
            }));
        }
        else {
            call_constructor(&self->scans, py::new_list(ScanND::alloc()));
        }
        if (nscans < 1)
            py_throw_format(PyExc_ValueError,
                            "Invalid serialization of ScanGroup: empty scans array.");
        call_constructor(&self->scanscache, py::new_nlist(nscans, [&] (auto i) {
            return py::new_none();
        }));
        return self;
    }
    static py::str_ref py_str(py::ptr<ScanGroup> self)
    {
        py::stringio io;
        io.write_ascii("ScanGroup\n");
        if (!self->base->is_default()) {
            io.write_ascii("  Scan Base:\n");
            self->base->show(io, 4, py::empty_tuple);
        }
        assert(self->scans.size() >= 1);
        if (self->scans.size() > 1 || !self->scans.get<ScanND>(0)->is_default()) {
            for (auto [i, scan]: py::list_iter<ScanND>(self->scans)) {
                io.write_ascii("  Scan ");
                io.write_cxx<32>(i);
                io.write_ascii(":\n");
                scan->show(io, 4, py::empty_tuple);
            }
        }
        return io.getvalue();
    }
    static py::ref<ScanGroup> cat_scan(py::ptr<> scan1, py::ptr<> scan2)
    {
        auto self = py::generic_alloc<ScanGroup>();
        call_constructor(&self->base, ScanND::alloc());
        call_constructor(&self->scans, py::new_list(0));
        call_constructor(&self->scanscache, py::new_list(0));
        // self->new_empty_called = false;
        self->add_cat_scan(scan1);
        self->add_cat_scan(scan2);
        return self;
    }

    static PyTypeObject Type;
};

struct ScanWrapper : PyObject {
    py::ref<ScanGroup> sg;
    py::ref<ScanND> scan;
    py::tuple_ref path;
    int idx;

    template<typename T>
    static py::ref<ScanWrapper> alloc(py::ptr<ScanGroup> sg, py::ptr<ScanND> scan,
                                      T &&path, int idx)
    {
        auto self = py::generic_alloc<ScanWrapper>();
        call_constructor(&self->sg, sg.ref());
        call_constructor(&self->scan, scan.ref());
        call_constructor(&self->path, py::newref(std::forward<T>(path)));
        self->idx = idx;
        *(void**)(self.get() + 1) = (void*)py::vectorfunc<vectorcall>;
        return self;
    }
    static void vectorcall(py::ptr<ScanWrapper> self, PyObject *const *args,
                           ssize_t nargs, py::tuple kwnames)
    {
        py::check_no_kwnames("ScanWrapper", kwnames);
        py::check_num_arg("ScanWrapper", nargs, 1, 2);
        py::tuple path = self->path;
        int pathlen = path.size();
        if (pathlen < 2 || path.get<py::str>(pathlen - 1).compare_ascii("scan") != 0)
            py_throw_format(PyExc_SyntaxError, "Invalid scan syntax");
        int idx;
        py::ptr<> v;
        if (nargs == 1) {
            idx = 0;
            v = args[0];
        }
        else {
            assert(nargs == 2);
            idx = py::arg_cast<py::int_>(args[0], "scanidx").as_int();
            if (idx < 0)
                py_throw_format(PyExc_IndexError,
                                "Scan dimension must not be negative: %d.", idx);
            v = args[1];
        }
        self->sg->set_scan_param(self->scan, py::new_ntuple(pathlen - 1, [&] (int i) {
            return path.get(i);
        }), self->idx, idx, v);
    }
    static py::str_ref py_str(py::ptr<ScanWrapper> self)
    {
        py::stringio io;
        if (self->idx == -1) {
            io.write_ascii("Scan Base");
        }
        else {
            io.write_ascii("Scan ");
            io.write_cxx<32>(self->idx);
        }
        if (self->path.size()) {
            io.write_ascii(" [");
            for (auto [i, p]: py::tuple_iter(self->path)) {
                io.write_ascii(".");
                io.write(p);
            }
            io.write_ascii("]");
        }
        io.write_ascii(":\n");
        self->scan->show(io, 2, self->path);
        return io.getvalue();
    }

    static PyTypeObject Type;
};
static auto ScanWrapper_as_number = PyNumberMethods{
    .nb_add = py::binfunc<[] (py::ptr<> self, py::ptr<> other) {
        return ScanGroup::cat_scan(self, other); }>,
};
static auto ScanWrapper_as_sequence = PySequenceMethods{
    .sq_ass_item = py::sq_ass_item<[] (py::ptr<ScanWrapper> self,
                                       Py_ssize_t idx, py::ptr<> v) {
        if (idx < 0)
            py_throw_format(PyExc_IndexError,
                            "Scan dimension must not be negative: %d.", idx);
        py::tuple path = self->path;
        int pathlen = path.size();
        if (pathlen < 2 || path.get<py::str>(pathlen - 1).compare_ascii("scan") != 0)
            py_throw_format(PyExc_SyntaxError, "Invalid scan syntax");
        self->sg->set_scan_param(self->scan, py::new_ntuple(pathlen - 1, [&] (int i) {
            return path.get(i);
        }), self->idx, idx, v);
    }>
};
PyTypeObject ScanWrapper::Type = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "brassboard_seq.scan.ScanWrapper",
    .tp_basicsize = sizeof(ScanWrapper) + sizeof(void*),
    .tp_dealloc = py::tp_cxx_dealloc<true,ScanWrapper>,
    .tp_vectorcall_offset = sizeof(ScanWrapper),
    .tp_repr = py::unifunc<py_str>,
    .tp_as_number = &ScanWrapper_as_number,
    .tp_as_sequence = &ScanWrapper_as_sequence,
    .tp_call = PyVectorcall_Call,
    .tp_str = py::unifunc<py_str>,
    .tp_getattro = py::binfunc<[] (py::ptr<ScanWrapper> self,
                                   py::str name) -> py::ref<> {
        py::check_non_empty_string(name, "name");
        if (PyUnicode_READ_CHAR(name, 0) == '_')
            return py::ref(PyObject_GenericGetAttr(self, name));
        return ScanWrapper::alloc(self->sg, self->scan, self->path.append(name), self->idx);
    }>,
    .tp_setattro = py::itrifunc<[] (py::ptr<ScanWrapper> self, py::str name,
                                    py::ptr<> value) {
        py::check_non_empty_string(name, "name");
        // To be consistent with __getattribute__
        if (PyUnicode_READ_CHAR(name, 0) == '_')
            py_throw_format(PyExc_AttributeError,
                            "'brassboard_seq.scan.ScanWrapper' object has no attribute '%U'", name);
        self->sg->set_fixed_param(self->scan, self->path.append(name), self->idx, value);
    }>,
    .tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_HAVE_GC|Py_TPFLAGS_HAVE_VECTORCALL,
    .tp_traverse = py::tp_field_traverse<ScanWrapper,&ScanWrapper::sg,&ScanWrapper::scan>,
    .tp_clear = py::tp_field_clear<ScanWrapper,&ScanWrapper::sg,&ScanWrapper::scan,&ScanWrapper::path>,
};

inline void ScanGroup::add_cat_scan(py::ptr<> scan)
{
    if (auto sw = py::exact_cast<ScanWrapper>(scan)) {
        if (sw->path != py::empty_tuple)
            py_throw_format(PyExc_ValueError, "Only top-level Scan can be concatenated.");
        add_cat_scannd(sw->sg, sw->idx);
    }
    else if (auto sg = py::exact_cast<ScanGroup>(scan)) {
        for (auto [idx, _]: py::list_iter(sg->scans)) {
            add_cat_scannd(sg, idx);
        }
    }
    else {
        py_throw_format(PyExc_TypeError, "Invalid type %S in scan concatenation.", scan.type());
    }
}

inline void ScanGroup::set_fixed_param(py::ptr<ScanND> scan, py::tuple path,
                                       int idx, py::ptr<> v)
{
    scan->check_noconflict(path, -1);
    if (v.typeis<ScanGroup>() || v.typeis<ScanWrapper>())
        py_throw_format(PyExc_TypeError, "Scan parameter cannot be a scan.");
    set_dirty(idx);
    check_param_overwrite(scan->fixed, path, v.isa<py::dict>());
    recursive_assign(scan->fixed, path, convert_param_value(v, true));
}

inline void ScanGroup::set_scan_param(py::ptr<ScanND> scan, py::tuple path, int idx,
                                      int scandim, py::ptr<> _v)
{
    if (_v.typeis<ScanGroup>() || _v.typeis<ScanWrapper>())
        py_throw_format(PyExc_TypeError, "Scan parameter cannot be a scan.");
    if (_v.typeis<py::dict>())
        py_throw_format(PyExc_TypeError, "Scan parameter cannot be a dict.");
    if (auto m = _v.type()->tp_as_sequence; !m || !m->sq_length)
        return set_fixed_param(scan, path, idx, _v);
    auto v = py::new_list(0);
    for (auto e: _v.generic_iter())
        v.append(convert_param_value(e, false));
    int nvals = v.size();
    if (!nvals)
        return;
    if (nvals == 1)
        return set_fixed_param(scan, path, idx, v.get(0));
    scan->check_noconflict(path, scandim);
    set_dirty(idx);
    py::list vars = scan->vars;
    for (int i = 0, n = scandim - vars.size() + 1; i < n; i++)
        vars.append(Scan1D::alloc());
    auto s1d = vars.get<Scan1D>(scandim);
    if (s1d->size != nvals) {
        if (s1d->size)
            py_throw_format(PyExc_ValueError, "Scan parameter size does not match.");
        s1d->size = nvals;
    }
    recursive_assign(s1d->params, path, v);
}

static auto ScanGroup_as_number = PyNumberMethods{
    .nb_add = py::binfunc<[] (py::ptr<> self, py::ptr<> other) {
        return ScanGroup::cat_scan(self, other); }>,
};
static auto ScanGroup_as_mapping = PyMappingMethods{
    .mp_subscript = py::binfunc<[] (py::ptr<ScanGroup> self, py::ptr<> _idx) {
        if (py::is_slice_none(_idx))
            return ScanWrapper::alloc(self, self->base, py::empty_tuple, -1);
        auto idx = py::arg_cast<py::int_>(_idx, "scanidx").as_int();
        py::list scans = self->scans;
        int nscans = scans.size();
        if (idx < 0) {
            idx += nscans;
            if (idx < 0) {
                py_throw_format(PyExc_IndexError,
                                "Scan group index out of bound: %d.", idx);
            }
        }
        else {
            self->add_empty(idx - nscans + 1);
        }
        return ScanWrapper::alloc(self, scans.get(idx), py::empty_tuple, idx);
    }>,
    .mp_ass_subscript = py::itrifunc<[] (py::ptr<ScanGroup> self,
                                         py::ptr<> _idx, py::ptr<> v) {
        py::ref<ScanND> new_scan;
        if (auto sw = py::exact_cast<ScanWrapper>(v)) {
            if (sw->sg != self)
                py_throw_format(PyExc_ValueError, "ScanGroup mismatch in assignment.");
            if (sw->path != py::empty_tuple)
                py_throw_format(PyExc_ValueError,
                                "Only top-level Scan can be assigned.");
            new_scan = sw->scan->copy();
        }
        else if (auto d = py::cast<py::dict>(v)) {
            new_scan = ScanND::alloc();
            new_scan->fixed = py::dict_deepcopy(d);
        }
        else {
            py_throw_format(PyExc_TypeError, "Invalid type %S in scan assignment.",
                            v.type());
        }
        if (py::is_slice_none(_idx)) {
            new_scan->baseidx = -1;
            self->base = std::move(new_scan);
        }
        else {
            int idx = py::arg_cast<py::int_>(_idx, "scanidx").as_int();
            py::list scans = self->scans;
            int nscans = scans.size();
            if (idx < 0) {
                idx += nscans;
                if (idx < 0) {
                    py_throw_format(PyExc_IndexError,
                                    "Scan group index out of bound: %d.", idx);
                }
            }
            else {
                self->add_empty(idx - nscans + 1);
            }
            scans.set(idx, std::move(new_scan));
        }
    }>,
};
PyTypeObject ScanGroup::Type = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "brassboard_seq.scan.ScanGroup",
    .tp_basicsize = sizeof(ScanGroup),
    .tp_dealloc = py::tp_cxx_dealloc<true,ScanGroup>,
    .tp_repr = py::unifunc<py_str>,
    .tp_as_number = &ScanGroup_as_number,
    .tp_as_mapping = &ScanGroup_as_mapping,
    .tp_str = py::unifunc<py_str>,
    .tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_HAVE_GC,
    .tp_doc = R"__pydoc__(
# Terminology:
* Parameter: a nested struct that will be passed to `Seq`
  as the context of a sequence.

* Scan: a n-dimensional matrix of parameters to iterate over with sequences.

  This is used to generate a list of parameters.
  A scan may contain some fixed parameters and some variable parameters.
  This is represented by a top-level `ScanWrapper`.

* Group: a (orderred) set of scans. Represented by `ScanGroup`.

* Fallback (parameter/scan):

  This contains the same information as a scan
  but it does not correspond to any real sequence.
  This contains the default fallback parameters for the real scans when the scan
  does not have any value for a specific field.

* Base index:

  This is the index of the scan that is used as fallback for this scan.
  If this is `-1`, the default one for the group is used.

Note: <> is used to indicate optional parameter below.

This class represents a group of scans. Supported API:
For sequence building:
* grp[:] / grp[:] = ...:
  grp[n] / grp[n] = ...:
    Access the group's fallback parameter (`grp[:]`) or
    the parameter for the n-th scan (`grp[n]`).
    `n` can use negative number syntax for the number of scans.

    Mutation to the fallback parameter will affect the fallback values of
    **ALL** scans in this `ScanGroup` including future ones.

    Read access returns a top-level `ScanWrapper`.
    Write access constructs a scan to **replace** the existing one.
    The (right hand side) RHS must be another top-level `ScanWrapper`
    from the same `ScanGroup` or a `dict`.

    For `ScanWrapper` RHS, everything is copied. Fallback values are not applied.
    If the LHS is `grp[:]`, base index of the RHS is ignored
    (otherwise, it is copied).
    For `dict` RHS, all fields are treated as non-scanning parameters.
    This (`struct` RHS) will also clear the scan
    and set the base index to `-1` (default).

* scans1 + scans2:
    Create a new `ScanGroup` that runs the individual all input scans
    in the order they are listed.
    Input could be either `ScanGroup` or top-level `ScanWrapper`
    created by indexing `ScanGroup`.
    `ScanWrapper` created with array indexing is supported in this case.
    The new group will **NOT** be affected if the inputs are mutated later.
    The scans will all have their respected fallback parameters merged into them
    and the base index reset to `-1`.
    The new group size (sequence count) will be the sum of that of the input
    groups and the order of the scans/sequences will be maintained.

* grp.setbaseidx(scan, base):
    Set the base index of the `scan` to `base`.
    A `-1` `base` means the default base.
    Throws an error if this would create a loop.

* grp.new_empty():
    Create a new empty scan in the group.
    The scan doesn't have any new parameters or scans set and
    has the base set to default.
    The index of the new scan is returned.
    If the first scan (which always exist after construction) is the only scan and
    it does not yet have any parameters set **and**
    this function has not been called, this function will not add a new scan
    and `0` is returned (i.e. as if the first scan didn't exist before).

    Together with the negative index,
    this allow adding scans in a group in the pattern,

        grp.new_empty();
        grp[-1]. .... = ...;
        grp[-1]. .... = ...;
        grp[-1]. .... = ...;

    which can be commented out as a whole easily.

All mutation on a scan (assignment and `setbaseidx`)
will make sure the scan being mutated is created if it didn't exist.


For sequence running/saving/loading:
* grp.groupsize():
    Number of scans in the group. (Each scan is a N-dimensional matrix)

* grp.scansize(idx):
    Number of sequences in the specific (N-dimensional) scan.

* grp.scandim(idx):
    Dimension of the scan. This includes dummy dimensions.

* grp.nseq():
    Number of sequences in the group. This is the sum of `scansize` over all scans.

* grp.getseq(n):
    Get the n-th sequence parameter.

* grp.getseq_in_scan(scan, n):
    Get the n-th sequence parameter within the scan-th scan.

* grp.dump():
    Return a low level dict data structure that can be saved without
    refering to any classes related to the scan.
    This can be later loaded to create an identical scan.
    If there are significant change on the representation of the scan,
    the new version is expected to load and convert
    the result generated by an older version without error.

* ScanGroup.load(obj):
    This is the reverse of `dump`. Returns a `ScanGroup` that is identical to the
    one that generates the representation with `dump`.

* grp.get_fixed(idx):
    Get the fixed (non-scan) parameters for a particular scan as a (nested) `dict`.

* grp.get_vars(idx, dim):
    Get the variable (scan) parameters for a particular scan along the scan
    dimension as a `dict`. Each non-scalar-struct field should be an array
    of the same length.
    The second return value is the size along the dimension, 0 size represents a
    dummy dimension that should be ignored.

* grp.axisnum(<idx<, dim>>):
    Get the number of scan parameters for the `idx`th scan in the group
    along the `dim`th axis. `idx` and `dim` both default to `0`.
    Return `0` for out-of-bound dimension. Error for out-of-bound scan
    index.
)__pydoc__",
    .tp_traverse = py::tp_field_traverse<ScanGroup,&ScanGroup::base,&ScanGroup::scans,
    &ScanGroup::scanscache>,
    .tp_clear = py::tp_field_clear<ScanGroup,&ScanGroup::base,&ScanGroup::scans,
    &ScanGroup::scanscache>,
    .tp_methods = (
        py::meth_table<
        py::meth_noargs<"new_empty",[] (py::ptr<ScanGroup> self) {
            int idx = self->scans.size();
            if (!self->new_empty_called && idx == 1 &&
                self->scans.get<ScanND>(0)->is_default()) {
                self->new_empty_called = true;
                return py::new_int(0);
            }
            self->add_empty();
            return py::new_int(idx);
        }>,
        py::meth_fast<"setbaseidx",[] (py::ptr<ScanGroup> self, PyObject *const *args,
                                       Py_ssize_t nargs) {
            py::check_num_arg("ScanGroup.setbaseidx", nargs, 2, 2);
            auto idx = py::arg_cast<py::int_>(args[0], "idx").as_int();
            auto base = py::arg_cast<py::int_>(args[1], "base").as_int();
            // This always makes sure that the scan we set the base for exists
            // and is initialized. The cache entry for this will also be
            // initialized. The caller might depend on this behavior.
            py::list scans = self->scans;
            int nscans = scans.size();
            if (base < -1)
                py_throw_format(PyExc_IndexError, "Invalid base index: %d.", base);
            if (base >= nscans)
                py_throw_format(PyExc_IndexError,
                                "Cannot set base to non-existing scan: %d.", base);
            if (idx >= nscans) {
                // New scan
                self->add_empty(idx - nscans + 1);
                scans.get<ScanND>(idx)->baseidx = base;
                return;
            }
            auto scan = scans.get<ScanND>(idx);
            // Fast pass to avoid invalidating anything
            if (scan->baseidx == base)
                return;
            if (base == -1) {
                // Set back to default, no possibility of new loop.
                self->set_dirty(idx);
                scan->baseidx = -1;
                return;
            }
            auto newbase = base;
            // Loop detection.
            std::vector<bool> visited(nscans, false);
            visited[idx] = true;
            while (true) {
                if (visited[base])
                    py_throw_format(PyExc_ValueError, "Base index loop detected.");
                visited[base] = true;
                base = scans.get<ScanND>(base)->baseidx;
                if (base == -1) {
                    break;
                }
            }
            scan->baseidx = newbase;
            self->set_dirty(idx);
        }>,
        py::meth_o<"getbaseidx",[] (py::ptr<ScanGroup> self, py::ptr<> _idx) {
            auto idx = py::arg_cast<py::int_>(_idx, "idx").as_int();
            int nscans = self->scans.size();
            if (idx < 0 || idx >= nscans)
                py_throw_format(PyExc_IndexError,
                                "Scan group index out of bound: %d.", idx);
            return py::new_int(self->scans.get<ScanND>(idx)->baseidx);
        }>,
        py::meth_noargs<"groupsize",[] (py::ptr<ScanGroup> self) {
            return py::new_int(self->scans.size());
        }>,
        py::meth_o<"scandim",[] (py::ptr<ScanGroup> self, py::ptr<> _idx) {
            auto idx = py::arg_cast<py::int_>(_idx, "idx").as_int();
            return py::new_int(self->getfullscan(idx, false)->vars.size());
        }>,
        py::meth_o<"scansize",[] (py::ptr<ScanGroup> self, py::ptr<> _idx) {
            auto idx = py::arg_cast<py::int_>(_idx, "idx").as_int();
            return py::new_int(self->getfullscan(idx, false)->size());
        }>,
        py::meth_fast<"axisnum",[] (py::ptr<ScanGroup> self, PyObject *const *args,
                                    Py_ssize_t nargs) {
            py::check_num_arg("ScanGroup.axisnum", nargs, 0, 2);
            int idx = 0;
            int dim = 0;
            switch (nargs) {
            case 2:
                dim = py::arg_cast<py::int_>(args[1], "dim").as_int();
                [[fallthrough]];
            case 1:
                idx = py::arg_cast<py::int_>(args[0], "scanidx").as_int();
                [[fallthrough]];
            default:
                break;
            }
            auto scan = self->getfullscan(idx, false);
            py::list vars = scan->vars;
            if (dim < 0 || dim >= vars.size())
                return py::new_int(0);
            auto var = vars.get<Scan1D>(dim);
            if (var->size == 0)
                return py::new_int(0);
            int res = 0;
            foreach_nondict(var->params, [&] (auto) { res += 1; });
            return py::new_int(res);
        }>,
        py::meth_noargs<"nseq",[] (py::ptr<ScanGroup> self) {
            int nscans = self->scans.size();
            int res = 0;
            for (int i = 0; i < nscans; i++)
                res += self->getfullscan(i, false)->size();
            return py::new_int(res);
        }>,
        py::meth_o<"getseq",[] (py::ptr<ScanGroup> self, py::ptr<> _n) {
            auto n = py::arg_cast<py::int_>(_n, "n").as_int();
            int nscans = self->scans.size();
            int orig_n = n;
            for (int scani = 0; scani < nscans; scani++) {
                auto scan = self->getfullscan(scani, false);
                auto ss = scan->size();
                if (n < ss)
                    return scan->getseq(n);
                n -= ss;
            }
            py_throw_format(PyExc_IndexError, "Sequence index out of bound: %d.", orig_n);
        }>,
        py::meth_fast<"getseq_in_scan",[] (py::ptr<ScanGroup> self, PyObject *const *args,
                                           Py_ssize_t nargs) {
            py::check_num_arg("ScanGroup.getseq_in_scan", nargs, 2, 2);
            auto scanidx = py::arg_cast<py::int_>(args[0], "scanidx").as_int();
            auto seqidx = py::arg_cast<py::int_>(args[1], "seqidx").as_int();
            return self->getfullscan(scanidx, false)->getseq(seqidx);
        }>,
        py::meth_o<"get_single_axis",[] (py::ptr<ScanGroup> self, py::ptr<> _scanidx) {
            auto scanidx = py::arg_cast<py::int_>(_scanidx, "scanidx").as_int();
            auto scan = self->getfullscan(scanidx, false);
            py::ptr<Scan1D> scanvar;
            for (auto [i, var]: py::list_iter<Scan1D>(scan->vars)) {
                if (var->size == 0)
                    continue;
                if (scanvar)
                    return py::new_tuple(py::new_none(), py::new_tuple());
                scanvar = var;
            }
            if (!scanvar)
                return py::new_tuple(py::new_none(), py::new_tuple());
            py::dict params = scanvar->params;
            py::tuple_ref path = py::new_tuple();
            while (true) {
                for (auto [k, v]: py::dict_iter(params)) {
                    path = path.append(k);
                    if (!v.typeis<py::dict>())
                        return py::new_tuple(v, std::move(path));
                    params = v;
                    break;
                }
            }
        }>,
        py::meth_o<"get_fixed",[] (py::ptr<ScanGroup> self, py::ptr<> _scanidx) {
            auto scanidx = py::arg_cast<py::int_>(_scanidx, "scanidx").as_int();
            return self->getfullscan(scanidx, false)->fixed.ref();
        }>,
        py::meth_fast<"get_vars",[] (py::ptr<ScanGroup> self, PyObject *const *args,
                                     Py_ssize_t nargs) {
            py::check_num_arg("ScanGroup.get_vars", nargs, 1, 2);
            int idx = py::arg_cast<py::int_>(args[0], "scanidx").as_int();
            int dim = 0;
            switch (nargs) {
            case 2:
                dim = py::arg_cast<py::int_>(args[1], "dim").as_int();
                [[fallthrough]];
            default:
                break;
            }
            py::list vars = self->getfullscan(idx, false)->vars;
            auto nvars = vars.size();
            if (dim < 0 || dim >= nvars)
                py_throw_format(PyExc_IndexError, "Scan dimension out of bound: %d.", dim);
            auto var = vars.get<Scan1D>(dim);
            return py::new_tuple(var->params, py::new_int(var->size));
        }>,
        py::meth_noargs<"dump",[] (py::ptr<ScanGroup> self) {
            auto res = py::new_dict();
            res.set("version"_py, py::int_cached(1));
            res.set("base"_py, self->base->dump(false));
            res.set("scans"_py, py::new_nlist(self->scans.size(), [&] (int i) {
                return self->scans.get<ScanND>(i)->dump(true);
            }));
            return res;
        }>,
        py::meth_o<"load",[] (auto, py::ptr<> _obj) {
            auto obj = py::arg_cast<py::dict>(_obj, "obj");
            auto _version = obj.try_get("version"_py);
            if (!_version)
                py_throw_format(PyExc_ValueError, "Version missing.");
            auto version = _version.as_int();
            if (version == 1)
                return load_v1(obj);
            py_throw_format(PyExc_ValueError, "Unsupported version: %d", version);
        },"",METH_STATIC>>),
    .tp_vectorcall = py::vectorfunc<[] (PyObject*, PyObject *const*,
                                        ssize_t nargs, py::tuple kwnames) {
        py::check_num_arg("ScanGroup.__init__", nargs, 0, 0);
        py::check_no_kwnames("ScanGroup.__init__", kwnames);
        auto self = py::generic_alloc<ScanGroup>();
        call_constructor(&self->base, ScanND::alloc());
        call_constructor(&self->scans, py::new_list(ScanND::alloc()));
        call_constructor(&self->scanscache, py::new_list(py::new_none()));
        return self;
    }>,
};

}

PyTypeObject &ScanGroup_Type = ScanGroup::Type;

__attribute__((visibility("hidden")))
void init()
{
    _import_array();
    init_parampack();
    throw_if(PyType_Ready(&Scan1D::Type) < 0);
    throw_if(PyType_Ready(&ScanND::Type) < 0);
    throw_if(PyType_Ready(&ScanWrapper::Type) < 0);
    throw_if(PyType_Ready(&ScanGroup::Type) < 0);
}

}
