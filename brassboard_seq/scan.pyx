# cython: language_level=3

# Copyright (c) 2024 - 2025 Yichao Yu <yyc1992@gmail.com>

# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 3.0 of the License, or (at your option) any later version.

# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public
# License along with this library. If not,
# see <http://www.gnu.org/licenses/>.

# Do not use relative import since it messes up cython file name tracking
from brassboard_seq.utils cimport PyErr_Format, PyExc_AttributeError, \
  PyExc_IndexError, PyExc_SyntaxError, PyExc_TypeError, \
  PyExc_ValueError, assume_not_none, _assume_not_none, pytuple_append1, \
  pydict_deepcopy, PyDict_GET_SIZE
from brassboard_seq.yaml cimport sprint as yaml_print

cdef StringIO # hide import
from io import StringIO

cdef np
import numpy as np

cimport cython

from cpython cimport PyObject, \
  PyDict_GetItemWithError, PyDictProxy_New, \
  PyTuple_GET_SIZE, PyTuple_GET_ITEM, \
  PyList_GET_SIZE, PyList_GET_ITEM, \
  PyUnicode_CompareWithASCIIString, PyObject_GenericGetAttr

from libcpp.vector cimport vector

cdef extern from "src/scan.cpp" namespace "brassboard_seq::scan":
    void merge_dict_into(object tgt, object src, bint ovr) except +
    dict ensure_visited(ParamPack self) except +
    dict ensure_dict(ParamPack self) except +
    object get_value(ParamPack self) except +
    object get_value_default(ParamPack self, object) except +
    object parampack_call(ParamPack self, tuple args, dict kwargs) except +

@cython.final
cdef class ParamPack:
    def __init__(self, *args, **kwargs):
        self.visited = {}
        self.fieldname = 'root'
        cdef int nargs = PyTuple_GET_SIZE(args)
        if PyDict_GET_SIZE(kwargs) == 0 and nargs == 0:
            self.values = kwargs # Use the "free" dict
            return
        self.values = {'root': kwargs}
        cdef int i
        cdef PyObject *argp
        for i in range(nargs):
            argp = PyTuple_GET_ITEM(args, nargs - 1 - i)
            if not isinstance(<object>argp, dict):
                PyErr_Format(PyExc_TypeError,
                             "Cannot use value as default value for parameter pack")
            merge_dict_into(kwargs, <dict>argp, True)

    def __contains__(self, str key):
        fieldname = self.fieldname
        values = self.values
        cdef PyObject *fieldp = PyDict_GetItemWithError(values, fieldname)
        if fieldp == NULL:
            return False
        field = <object>fieldp
        if not isinstance(field, dict):
            PyErr_Format(PyExc_TypeError, "Scalar value does not have field")
        return key in <dict>field

    def __getitem__(self, key):
        if key != slice(None):
            PyErr_Format(PyExc_ValueError,
                         "Invalid index for ParamPack: %S", <PyObject*>key)
        fieldname = self.fieldname
        values = self.values
        cdef PyObject *fieldp = PyDict_GetItemWithError(values, fieldname)
        if fieldp == NULL:
            return {}
        field = <object>fieldp
        if not isinstance(field, dict):
            PyErr_Format(PyExc_TypeError, "Cannot access value as parameter pack.")
        return {k: pydict_deepcopy(v) for k, v in (<dict>field).items()}

    def __getattribute__(self, str name):
        assume_not_none(name)
        if name.startswith('_'):
            return PyObject_GenericGetAttr(self, name)
        return new_param_pack(ParamPack, ensure_dict(self), ensure_visited(self),
                              name, None)

    def __setattr__(self, str name, value):
        assume_not_none(name)
        if name.startswith('_'):
            # To be consistent with __getattribute__
            PyErr_Format(PyExc_AttributeError,
                         "'ParamPack' object has no attribute '%U'", <PyObject*>name)
        self_values = ensure_dict(self)
        cdef PyObject *oldvaluep = PyDict_GetItemWithError(self_values, name)
        if oldvaluep != NULL:
            was_dict = isinstance(<object>oldvaluep, dict)
            is_dict = isinstance(value, dict)
            if was_dict and not is_dict:
                PyErr_Format(PyExc_TypeError, "Cannot override parameter pack as value")
            if not was_dict and is_dict:
                PyErr_Format(PyExc_TypeError, "Cannot override value as parameter pack")
            if is_dict:
                merge_dict_into(<dict>oldvaluep, <dict>value, True)
            else:
                assume_not_none(self_values)
                self_values[name] = value
        else:
            assume_not_none(self_values)
            self_values[name] = pydict_deepcopy(value)

    def __call__(self, *args, **kwargs):
        # Supported syntax
        # () -> get value without default
        # (value) -> get value with default
        # (*dicts, **kwargs) -> get parameter pack with default
        return parampack_call(self, args, kwargs)

    def __str__(self):
        fieldname = self.fieldname
        values = self.values
        cdef PyObject *fieldp = PyDict_GetItemWithError(values, fieldname)
        if fieldp == NULL:
            return '<Undefined>'
        field = <object>fieldp
        if not isinstance(field, dict):
            return str(field)
        return yaml_print(field)

    def __repr__(self):
        return str(self)

def get_visited(ParamPack self, /):
    fieldname = self.fieldname
    visited = self.visited
    cdef PyObject *resp = PyDict_GetItemWithError(visited, fieldname)
    if resp != NULL:
        res = <object>resp
        if isinstance(res, dict):
            return PyDictProxy_New(res)
        return res
    if isinstance(self.values.get(fieldname), dict):
        res = {}
        assume_not_none(visited)
        visited[fieldname] = res
        return PyDictProxy_New(res)
    return False

# Helper function for functions that takes an optional parameter pack
def get_param(param, /):
    if param is None:
        return new_param_pack(ParamPack, {}, {}, 'root', None)
    return param

# Check if the struct field reference path is overwritten in `obj`.
# Overwrite happens if the field itself exists or a parent of the field
# is overwritten to something that's not scalar struct.
cdef bint check_field(dict d, tuple path) except -1:
    cdef PyObject *vp
    cdef int pathlen = PyTuple_GET_SIZE(path)
    cdef int i
    for i in range(pathlen):
        vp = PyDict_GetItemWithError(d, <object>PyTuple_GET_ITEM(path, i))
        if vp == NULL:
            return False
        if not isinstance(<object>vp, dict):
            return True
        d = <dict>vp
    return True

cdef dict _missing_value = {}
cdef recursive_get(dict d, tuple path):
    cdef int pathlen = PyTuple_GET_SIZE(path)
    if not pathlen:
        return d
    cdef int i = 0
    for i in range(pathlen - 1):
        f = PyTuple_GET_ITEM(path, i)
        vp = PyDict_GetItemWithError(d, <object>f)
        if vp == NULL or not isinstance(<object>vp, dict):
            return _missing_value
        d = <dict>vp
    vp = PyDict_GetItemWithError(d, <object>PyTuple_GET_ITEM(path, pathlen - 1))
    if vp == NULL:
        return _missing_value
    return <object>vp

cdef int recursive_assign(dict d, tuple path, v) except -1:
    cdef int i = 0
    cdef int pathlen = PyTuple_GET_SIZE(path)
    assert pathlen > 0
    for i in range(pathlen - 1):
        f = PyTuple_GET_ITEM(path, i)
        vp = PyDict_GetItemWithError(d, <object>f)
        if vp == NULL or not isinstance(<object>vp, dict):
            newd = {}
            assume_not_none(d)
            d[<object>f] = newd
            d = newd
            continue
        d = <dict>vp
    assume_not_none(d)
    d[<object>PyTuple_GET_ITEM(path, pathlen - 1)] = v

ctypedef int (*foreach_nondict_cb)(object, tuple, void*) except -1

cdef int foreach_nondict(foreach_nondict_cb cb, void *data, dict obj) except -1:
    _foreach_nondict(cb, data, obj, ())

cdef int _foreach_nondict(foreach_nondict_cb cb, void *data, dict obj,
                          tuple prefix) except -1:
    assume_not_none(obj)
    for k, v in obj.items():
        path = pytuple_append1(prefix, k)
        if isinstance(v, dict):
            _foreach_nondict(cb, data, <dict>v, path)
        else:
            cb(v, path, data)


@cython.internal
@cython.final
cdef class Scan1D:
    cdef int size
    cdef dict params

cdef Scan1D new_scan1d():
    self = <Scan1D>Scan1D.__new__(Scan1D)
    self.size = 0
    self.params = {}
    return self

cdef Scan1D copy_scan1d(Scan1D self):
    copy = <Scan1D>Scan1D.__new__(Scan1D)
    copy.size = self.size
    copy.params = pydict_deepcopy(self.params)
    return copy

cdef dict dump_scan1d(Scan1D self):
    return dict(size=self.size, params=self.params)

cdef int load_scan1d_checksize_cb(v, tuple path, void *p) except -1:
    cdef int size = <int><long>p
    if not isinstance(v, list):
        T = type(v)
        PyErr_Format(PyExc_TypeError,
                     "Invalid serialization of ScanGroup: wrong parameter type %S.",
                     <PyObject*>T)
    if PyList_GET_SIZE(v) != size:
        PyErr_Format(PyExc_ValueError,
                     "Invalid serialization of ScanGroup: scan size mismatch, expect %d, got %d.",
                     size, <int>PyList_GET_SIZE(v))

cdef Scan1D load_scan1d(dict data):
    self = <Scan1D>Scan1D.__new__(Scan1D)
    self.size = data.get('size', 0)
    params = data.get('params', None)
    if params is None:
        self.params = {}
    else:
        self.params = pydict_deepcopy(<dict?>params)
        foreach_nondict(load_scan1d_checksize_cb, <void*><long>self.size, self.params)
    return self


@cython.internal
@cython.final
cdef class ScanND:
    cdef int baseidx
    cdef dict fixed
    cdef list vars


cdef ScanND new_scannd():
    self = <ScanND>ScanND.__new__(ScanND)
    self.baseidx = -1
    self.fixed = {}
    self.vars = []
    return self

cdef inline bint scannd_is_default(ScanND self) except -1:
    return (self.baseidx == -1 and not PyDict_GET_SIZE(self.fixed) and
            not PyList_GET_SIZE(self.vars))

cdef bint contains_path(ScanND self, tuple path) except -1:
    if check_field(self.fixed, path):
        return True
    vars = self.vars
    for i in range(PyList_GET_SIZE(vars)):
        if check_field((<Scan1D>PyList_GET_ITEM(vars, i)).params, path):
            return True
    return False

cdef ScanND copy_scannd(ScanND self):
    copy = <ScanND>ScanND.__new__(ScanND)
    copy.baseidx = self.baseidx
    copy.fixed = pydict_deepcopy(self.fixed)
    copy.vars = []
    _assume_not_none(<void*>self.vars)
    for scan1d in self.vars:
        copy.vars.append(copy_scan1d(<Scan1D>scan1d))
    return copy

cdef dict dump_scannd(ScanND self, bint dumpbase):
    res = dict(params=self.fixed, vars=[dump_scan1d(<Scan1D>var) for var in self.vars])
    if dumpbase:
        assume_not_none(res)
        res['baseidx'] = self.baseidx + 1
    return res

cdef ScanND load_scannd(dict data):
    self = <ScanND>ScanND.__new__(ScanND)
    self.baseidx = data.get('baseidx', 0) - 1
    fixed = data.get('params', None)
    if fixed is None:
        self.fixed = {}
    else:
        self.fixed = pydict_deepcopy(fixed)
    vars = data.get('vars', None)
    if vars is None:
        self.vars = []
    else:
        self.vars = [load_scan1d(v) for v in vars]
    return self

cdef int scannd_size(ScanND scan) noexcept:
    cdef int res = 1
    cdef int sz1d
    vars = scan.vars
    for i in range(PyList_GET_SIZE(vars)):
        sz1d = (<Scan1D>PyList_GET_ITEM(vars, i)).size
        if sz1d != 0:
            res *= sz1d
    return res

cdef struct scannd_getseq_setparam_data:
    PyObject *seq
    int subidx

cdef int scannd_getseq_setparam_cb(v, tuple path, void *p) except -1:
    cdef scannd_getseq_setparam_data *data = <scannd_getseq_setparam_data*>p
    assume_not_none(v)
    recursive_assign(<dict>data.seq, path, (<list>v)[data.subidx])

@cython.cdivision(True)
cdef dict scannd_getseq(ScanND scan, int seqidx):
    cdef int orig_idx = seqidx
    seq = <dict>pydict_deepcopy(scan.fixed)
    cdef scannd_getseq_setparam_data data
    data.seq = <PyObject*>seq
    cdef int subidx
    for i in range(PyList_GET_SIZE(scan.vars)):
        _assume_not_none(<void*>scan.vars)
        var = <Scan1D>PyList_GET_ITEM(scan.vars, i)
        if var.size == 0:
            continue
        subidx = seqidx % var.size
        seqidx = (seqidx - subidx) // var.size

        data.subidx = subidx
        foreach_nondict(scannd_getseq_setparam_cb, &data, var.params)
    if seqidx != 0:
        PyErr_Format(PyExc_IndexError, "Sequence index out of bound: %d.", orig_idx)
    return seq


@cython.c_api_binop_methods(True)
@cython.final
cdef class ScanWrapper:
    cdef ScanGroup sg
    cdef ScanND scan
    cdef tuple path
    cdef int idx

    def __getattribute__(self, str name):
        assume_not_none(name)
        if name.startswith('_'):
            return PyObject_GenericGetAttr(self, name)
        return new_scan_wrapper(self.sg, self.scan,
                                pytuple_append1(self.path, name), self.idx)

    def __setitem__(self, int idx, v):
        if idx < 0:
            PyErr_Format(PyExc_IndexError,
                         "Scan dimension must not be negative: %d.", idx)
        path = self.path
        cdef int pathlen = PyTuple_GET_SIZE(path)
        if pathlen < 2 or PyUnicode_CompareWithASCIIString(<str>PyTuple_GET_ITEM(path, pathlen - 1), 'scan') != 0:
            PyErr_Format(PyExc_SyntaxError, "Invalid scan syntax")
        set_scan_param(self.sg, self.scan, path[:-1], self.idx, idx, v)

    def __call__(self, *args):
        path = self.path
        cdef int pathlen = PyTuple_GET_SIZE(path)
        if pathlen < 2 or PyUnicode_CompareWithASCIIString(<str>PyTuple_GET_ITEM(path, pathlen - 1), 'scan') != 0:
            PyErr_Format(PyExc_SyntaxError, "Invalid scan syntax")
        cdef int nargs = PyTuple_GET_SIZE(args)
        cdef int idx
        if nargs == 1:
            v = <object>PyTuple_GET_ITEM(args, 0)
            idx = 0
        elif nargs == 2:
            idx = <int?><object>PyTuple_GET_ITEM(args, 0)
            v = <object>PyTuple_GET_ITEM(args, 1)
            if idx < 0:
                PyErr_Format(PyExc_IndexError,
                             "Scan dimension must not be negative: %d.", idx)
        else:
            PyErr_Format(PyExc_TypeError,
                         "Scan syntax takes 1 or 2 arguments, but %d were given",
                         nargs)
        set_scan_param(self.sg, self.scan, self.path[:-1], self.idx, idx, v)

    def __setattr__(self, str name, v):
        assume_not_none(name)
        if name.startswith('_'):
            # To be consistent with __getattribute__
            PyErr_Format(PyExc_AttributeError,
                         "'brassboard_seq.scan.ScanWrapper' object has no attribute '%U'", <PyObject*>name)
        set_fixed_param(self.sg, self.scan, pytuple_append1(self.path, name),
                        self.idx, v)

    def __add__(self, other):
        return cat_scan(self, other)

    def __str__(self):
        io = StringIO()
        write = io.write
        if PyTuple_GET_SIZE(self.path):
            path_str = ".".join(self.path)
            if self.idx == -1:
                write(f'Scan Base [.{path_str}]:\n')
            else:
                write(f'Scan {self.idx} [.{path_str}]:\n')
        else:
            if self.idx == -1:
                write('Scan Base:\n')
            else:
                write(f'Scan {self.idx}:\n')
        print_scan(self.scan, write, 2, self.path)
        return io.getvalue()

    def __repr__(self):
        return str(self)


cdef ScanWrapper new_scan_wrapper(ScanGroup sg, ScanND scan, tuple path, int idx):
    self = <ScanWrapper>ScanWrapper.__new__(ScanWrapper)
    self.sg = sg
    self.scan = scan
    self.path = path
    self.idx = idx
    return self

cdef inline int add_empty_scan(ScanGroup self, list scans) except -1:
    assume_not_none(scans)
    scans.append(new_scannd())
    _assume_not_none(<void*>self.scanscache)
    self.scanscache.append(None)

cdef inline int add_n_empty_scan(ScanGroup self, list scans, int n) except -1:
    for _ in range(n):
        add_empty_scan(self, scans)


@cython.c_api_binop_methods(True)
@cython.final
cdef class ScanGroup:
    """
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
    """
    cdef ScanND base
    cdef list scans
    cdef list scanscache # always the same size as scans
    cdef bint new_empty_called

    def __init__(self):
        self.base = new_scannd()
        self.scans = [new_scannd()]
        self.scanscache = [None]
        self.new_empty_called = False

    def new_empty(self):
        scans = self.scans
        cdef int idx = PyList_GET_SIZE(scans)
        if (not self.new_empty_called and idx == 1 and
            scannd_is_default(<ScanND>PyList_GET_ITEM(scans, 0))):
            self.new_empty_called = True
            return 0
        add_empty_scan(self, scans)
        return idx

    def __getitem__(self, _idx):
        if _idx == slice(None):
            return new_scan_wrapper(self, self.base, (), -1)
        cdef int idx = <int?>_idx
        scans = self.scans
        cdef int nscans = PyList_GET_SIZE(scans)
        if idx < 0:
            idx += nscans
            if idx < 0:
                PyErr_Format(PyExc_IndexError,
                             "Scan group index out of bound: %d.", idx)
        else:
            add_n_empty_scan(self, scans, idx - nscans + 1)
        return new_scan_wrapper(self, <ScanND>PyList_GET_ITEM(scans, idx), (), idx)

    def __setitem__(self, _idx, v):
        if isinstance(v, ScanWrapper):
            if (<ScanWrapper>v).sg is not self:
                PyErr_Format(PyExc_ValueError, "ScanGroup mismatch in assignment.")
            new_scan = copy_scannd((<ScanWrapper>v).scan)
        elif isinstance(v, dict):
            new_scan = new_scannd()
            new_scan.fixed = pydict_deepcopy(<dict>v)
        else:
            T = type(v)
            PyErr_Format(PyExc_TypeError, "Invalid type %S in scan assignment.",
                         <PyObject*>T)
        cdef int idx
        cdef int nscans
        if _idx == slice(None):
            new_scan.baseidx = -1
            self.base = new_scan
        else:
            idx = <int?>_idx
            scans = self.scans
            nscans = PyList_GET_SIZE(scans)
            if idx < 0:
                idx += nscans
                if idx < 0:
                    PyErr_Format(PyExc_IndexError,
                                 "Scan group index out of bound: %d.", idx)
            else:
                add_n_empty_scan(self, scans, idx - nscans + 1)
            _assume_not_none(<void*>scans)
            scans[idx] = new_scan

    def setbaseidx(self, int idx, int base):
        # This always makes sure that the scan we set the base for exists
        # and is initialized. The cache entry for this will also be initialized.
        # The caller might depend on this behavior.
        scans = self.scans
        cdef int nscans = PyList_GET_SIZE(scans)
        if base < -1:
            PyErr_Format(PyExc_IndexError, "Invalid base index: %d.", base)
        elif base >= nscans:
            PyErr_Format(PyExc_IndexError,
                         "Cannot set base to non-existing scan: %d.", base)
        elif idx >= nscans:
            # New scan
            add_n_empty_scan(self, scans, idx - nscans + 1)
            (<ScanND>PyList_GET_ITEM(scans, idx)).baseidx = base
            return
        scan = <ScanND>PyList_GET_ITEM(scans, idx)
        # Fast pass to avoid invalidating anything
        if scan.baseidx == base:
            return
        if base == -1:
            # Set back to default, no possibility of new loop.
            set_dirty(self, idx)
            scan.baseidx = -1
            return
        cdef int newbase = base
        # Loop detection.
        cdef vector[bint] visited = vector[bint](nscans, False)
        visited[idx] = True
        while True:
            if visited[base]:
                PyErr_Format(PyExc_ValueError, "Base index loop detected.")
            visited[base] = True
            base = (<ScanND>PyList_GET_ITEM(scans, base)).baseidx
            if base == -1:
                break
        scan.baseidx = newbase
        set_dirty(self, idx)

    def __str__(self):
        io = StringIO()
        write = io.write
        write('ScanGroup\n')
        if not scannd_is_default(self.base):
            write('  Scan Base:\n')
            print_scan(self.base, write, 4, ())
        if len(self.scans) > 1 or not scannd_is_default(<ScanND>self.scans[0]):
            for i in range(len(self.scans)):
                write(f'  Scan {i}:\n')
                print_scan(<ScanND>self.scans[i], write, 4, ())
        return io.getvalue()

    def __repr__(self):
        return str(self)

    def groupsize(self):
        return PyList_GET_SIZE(self.scans)

    def scansize(self, int idx):
        return scannd_size(getfullscan(self, idx, False))

    def scandim(self, int idx):
        scan = getfullscan(self, idx, False)
        return PyList_GET_SIZE(scan.vars)

    def axisnum(self, int idx=0, int dim=0):
        scan = getfullscan(self, idx, False)
        vars = scan.vars
        if (dim < 0 or dim >= PyList_GET_SIZE(vars) or
            (<Scan1D>PyList_GET_ITEM(vars, dim)).size == 0):
            return 0
        cdef int res = 0
        foreach_nondict(axisnum_counter_cb, &res,
                        (<Scan1D>PyList_GET_ITEM(vars, dim)).params)
        return res

    def getbaseidx(self, int idx):
        scans = self.scans
        nscans = PyList_GET_SIZE(scans)
        if idx < 0 or idx >= nscans:
            PyErr_Format(PyExc_IndexError,
                         "Scan group index out of bound: %d.", idx)
        return (<ScanND>PyList_GET_ITEM(scans, idx)).baseidx

    def nseq(self):
        cdef int res = 0
        cdef int i
        for i in range(PyList_GET_SIZE(self.scans)):
            res += scannd_size(getfullscan(self, i, False))
        return res

    def getseq(self, int n):
        cdef int scani
        cdef int ss
        cdef int orig_n = n
        for scani in range(PyList_GET_SIZE(self.scans)):
            scan = getfullscan(self, scani, False)
            ss = scannd_size(scan)
            if n < ss:
                return scannd_getseq(scan, n)
            n = n - ss
        PyErr_Format(PyExc_IndexError, "Sequence index out of bound: %d.", orig_n)

    def getseq_in_scan(self, int scanidx, int seqidx):
        return scannd_getseq(getfullscan(self, scanidx, False), seqidx)

    def get_single_axis(self, int scanidx):
        scan = getfullscan(self, scanidx, False)
        cdef Scan1D scanvar = None
        cdef int scandim = -1
        for i in range(PyList_GET_SIZE(scan.vars)):
            var = <Scan1D>PyList_GET_ITEM(scan.vars, i)
            if var.size == 0:
                continue
            if scanvar is not None:
                return None, ()
            scanvar = var
        if scanvar is None:
            return None, ()
        cdef dict params = scanvar.params
        cdef tuple path = ()
        while True:
            assume_not_none(params)
            for k, v in params.items():
                path = pytuple_append1(path, k)
                if not isinstance(v, dict):
                    return v, path
                params = <dict>v
                break

    def get_fixed(self, int idx):
        return getfullscan(self, idx, False).fixed

    def get_vars(self, int idx, int dim=0):
        vars = getfullscan(self, idx, False).vars
        nvars = PyList_GET_SIZE(vars)
        if dim < 0 or dim >= nvars:
            PyErr_Format(PyExc_IndexError, "Scan dimension out of bound: %d.", dim)
        var = <Scan1D>PyList_GET_ITEM(vars, dim)
        return var.params, var.size

    def dump(self):
        return dict(version=1, base=dump_scannd(self.base, False),
                    scans=[dump_scannd(<ScanND>scan, True) for scan in self.scans])

    @staticmethod
    def load(dict obj):
        version = obj.get('version', None)
        if version is None:
            PyErr_Format(PyExc_ValueError, "Version missing.")
        elif version == 1:
            return load_scangroup_v1(obj)
        else:
            PyErr_Format(PyExc_ValueError, "Unsupported version: %S",
                         <PyObject*>version)

    def __add__(self, other):
        return cat_scan(self, other)

cdef int axisnum_counter_cb(v, tuple path, void *_p) except -1:
    cdef int *p = <int*>_p
    p[0] += 1

cdef ScanGroup cat_scan(scan1, scan2):
    self = <ScanGroup>ScanGroup.__new__(ScanGroup)
    self.base = new_scannd()
    self.scans = []
    self.scanscache = []
    self.new_empty_called = False
    add_cat_scan(self, scan1)
    add_cat_scan(self, scan2)
    return self

cdef int add_cat_scannd(ScanGroup self, ScanGroup other, int idx) except -1:
    scan = copy_scannd(getfullscan(other, idx, True))
    scan.baseidx = -1
    _assume_not_none(<void*>self.scans)
    self.scans.append(scan)
    _assume_not_none(<void*>self.scanscache)
    self.scanscache.append(None)

cdef int add_cat_scan(ScanGroup self, scan) except -1:
    cdef int idx
    if isinstance(scan, ScanWrapper):
        sw = <ScanWrapper>scan
        if PyTuple_GET_SIZE(sw.path):
            PyErr_Format(PyExc_ValueError, "Only top-level Scan can be concatenated.")
        add_cat_scannd(self, sw.sg, sw.idx)
    elif isinstance(scan, ScanGroup):
        sg = <ScanGroup>scan
        for idx in range(PyList_GET_SIZE(sg.scans)):
            add_cat_scannd(self, sg, idx)
    else:
        T = type(scan)
        PyErr_Format(PyExc_TypeError, "Invalid type %S in scan concatenation.",
                     <PyObject*>T)

cdef load_scangroup_v1(dict data):
    self = <ScanGroup>ScanGroup.__new__(ScanGroup)
    self.new_empty_called = False
    base = data.get('base', None)
    if base is None:
        self.base = new_scannd()
    else:
        self.base = load_scannd(base)
    scans = data.get('scans', None)
    if scans is None:
        self.scans = [new_scannd()]
    else:
        self.scans = [load_scannd(v) for v in scans]
    cdef int nscans = PyList_GET_SIZE(self.scans)
    if nscans < 1:
        PyErr_Format(PyExc_ValueError,
                     "Invalid serialization of ScanGroup: empty scans array.")
    self.scanscache = [None for _ in range(nscans)]
    return self

cdef int print_scan(ScanND scan, write, int indent, tuple path) except -1:
    prefix = ' ' * indent
    empty = True
    if scan.baseidx != -1:
        empty = False
        write(prefix)
        write(f'Base index: {scan.baseidx}\n')
    fixed = recursive_get(scan.fixed, path)
    new_prefix = prefix + '   '
    if fixed:
        empty = False
        write(prefix)
        write('Fixed parameters:\n');
        write(new_prefix)
        write(yaml_print(fixed, indent + 3))
        write('\n')
    for i in range(len(scan.vars)):
        var = <Scan1D>scan.vars[i]
        if var.size == 0:
            continue
        empty = False
        write(prefix)
        write(f'Scan dimension {i}: (size {var.size})\n')
        params = recursive_get(var.params, path)
        write(new_prefix)
        if params is _missing_value:
            write('<empty>\n')
        else:
            write(yaml_print(params, indent + 3))
            write('\n')
    if empty:
        write(prefix)
        write('<empty>\n')

cdef inline int set_dirty(ScanGroup sg, int idx) except -1:
    scanscache = sg.scanscache
    cdef int i
    if idx < 0:
        for i in range(PyList_GET_SIZE(scanscache)):
            assume_not_none(scanscache)
            scanscache[i] = None
    else:
        assume_not_none(scanscache)
        scanscache[idx] = None

cdef int check_noconflict(ScanND scan, tuple path, int scandim) except -1:
    if scandim != -1:
        if check_field(scan.fixed, path):
            PyErr_Format(PyExc_ValueError, "Cannot scan a fixed parameter.")
    cdef int i
    vars = scan.vars
    for i in range(PyList_GET_SIZE(vars)):
        if i == scandim:
            continue
        if check_field((<Scan1D>PyList_GET_ITEM(vars, i)).params, path):
            if scandim == -1:
                PyErr_Format(PyExc_ValueError, "Cannot fix a scanned parameter.")
            else:
                PyErr_Format(PyExc_ValueError, "Cannot scan a parameter in multiple dimensions.")

# Check if the assigning `obj` with subfield reference `path`
# will cause forbidden overwrite. This happens when `path` exists in `obj`
# and if either the original value of the new value is a scalar struct.
# These assignments are forbidding since we do not want any field to
# change type (struct -> non-struct or non-struct -> struct).
# We also don't allow assigning struct to struct even if there's no conflict
# in the structure of the old and new values since the old value might have
# individual fields explicitly assigned previously and this has caused a few
# surprises in practice...
# We could in principle change the semantics to treat the assigned value as default
# value (so it will not overwrite any existing fields)
# but that is not super consistent with other semantics.
# If the more restricted semantics (error on struct assignment to struct)
# is proven insufficient, we can always add the merge semantics later without breaking.
cdef int check_param_overwrite(dict obj, tuple path, bint isdict) except -1:
    cdef int pathlen = PyTuple_GET_SIZE(path)
    for i in range(pathlen - 1):
        vp = PyDict_GetItemWithError(obj, <object>PyTuple_GET_ITEM(path, i))
        if vp == NULL:
            return 0
        if not isinstance(<object>vp, dict):
            PyErr_Format(PyExc_TypeError, "Assignment to field of scalar not allowed.")
        obj = <dict>vp

    vp = PyDict_GetItemWithError(obj, <object>PyTuple_GET_ITEM(path, pathlen - 1))
    if vp == NULL:
        return 0
    cdef bint wasdict = isinstance(<object>vp, dict)
    if isdict and not wasdict:
        PyErr_Format(PyExc_TypeError,
                     "Changing field from non-dict to dict not allowed.")
    elif not isdict and wasdict:
        PyErr_Format(PyExc_TypeError,
                     "Changing field from dict to non-dict not allowed.")
    elif isdict and wasdict:
        # See comment above for explaination.
        PyErr_Format(PyExc_TypeError, "Override dict not allowed.")

cdef np_bool = np.bool_
cdef tuple np_int_types = (np.int8, np.int16, np.int32, np.int64,
                           np.uint8, np.uint16, np.uint32, np.uint64)
cdef tuple np_float_types = (np.float16, np.float32, np.float64)
cdef np_array = np.ndarray

cdef convert_param_value(obj, bint allow_dict):
    if isinstance(obj, np_bool):
        return bool(obj)
    if isinstance(obj, np_int_types):
        return int(obj)
    if isinstance(obj, np_float_types):
        return float(obj)
    if isinstance(obj, list):
        return [convert_param_value(v, True) for v in <list>obj]
    if isinstance(obj, np_array):
        return [convert_param_value(v, True) for v in obj]
    if isinstance(obj, dict):
        if not allow_dict:
            PyErr_Format(PyExc_TypeError, "Scan parameter cannot be a dict.")
        return {k: convert_param_value(v, True) for k, v in (<dict>obj).items()}
    return obj

cdef int set_fixed_param(ScanGroup sg, ScanND scan, tuple path, int idx, v) except -1:
    check_noconflict(scan, path, -1)
    if isinstance(v, ScanGroup) or isinstance(v, ScanWrapper):
        PyErr_Format(PyExc_TypeError, "Scan parameter cannot be a scan.")
    set_dirty(sg, idx)
    check_param_overwrite(scan.fixed, path, isinstance(v, dict))
    recursive_assign(scan.fixed, path, convert_param_value(v, True))

cdef int set_scan_param(ScanGroup sg, ScanND scan, tuple path, int idx,
                        int scandim, _v) except -1:
    if isinstance(_v, ScanGroup) or isinstance(_v, ScanWrapper):
        PyErr_Format(PyExc_TypeError, "Scan parameter cannot be a scan.")
    if not hasattr(_v, '__len__'):
        return set_fixed_param(sg, scan, path, idx, _v)
    if isinstance(_v, dict):
        PyErr_Format(PyExc_TypeError, "Scan parameter cannot be a dict.")
    cdef list v = [convert_param_value(e, False) for e in _v]
    cdef int nvals = len(v)
    if nvals == 0:
        return 0
    if nvals == 1:
        return set_fixed_param(sg, scan, path, idx, <object>PyList_GET_ITEM(v, 0))
    check_noconflict(scan, path, scandim)
    set_dirty(sg, idx)
    vars = scan.vars
    for _ in range(scandim - PyList_GET_SIZE(vars) + 1):
        assume_not_none(vars)
        vars.append(new_scan1d())
    s1d = <Scan1D>PyList_GET_ITEM(vars, scandim)
    if s1d.size == 0:
        s1d.size = nvals
    elif s1d.size != nvals:
        PyErr_Format(PyExc_ValueError, "Scan parameter size does not match.")
    recursive_assign(s1d.params, path, v)

cdef struct getfullscan_cbs_data:
    PyObject *scan
    int idxsz

cdef int getfullscan_param_cb(v, tuple path, void *p) except -1:
    cdef getfullscan_cbs_data *data = <getfullscan_cbs_data*>p
    scan = <ScanND>data.scan
    if contains_path(scan, path):
        return 0
    recursive_assign(scan.fixed, path, v)

cdef int getfullscan_var_cb(v, tuple path, void *p) except -1:
    cdef getfullscan_cbs_data *data = <getfullscan_cbs_data*>p
    scan = <ScanND>data.scan
    if contains_path(scan, path):
        return 0
    cdef int scanid = data.idxsz
    vars = scan.vars
    for _ in range(scanid - PyList_GET_SIZE(vars) + 1):
        assume_not_none(vars)
        vars.append(new_scan1d())
    recursive_assign((<Scan1D>PyList_GET_ITEM(vars, scanid)).params, path, v)

cdef int getfullscan_count_vars_cb(v, tuple path, void *p) except -1:
    cdef getfullscan_cbs_data *data = <getfullscan_cbs_data*>p
    nv = PyList_GET_SIZE(v)
    assert nv > 1
    if data.idxsz == 0:
        data.idxsz = nv
    elif data.idxsz != nv:
        PyErr_Format(PyExc_ValueError, "Scan parameter size does not match.")

cdef ScanND getfullscan(ScanGroup self, int idx, bint allow_base):
    if idx == -1 and allow_base:
        return self.base
    scanscache = self.scanscache
    cdef int nscans = PyList_GET_SIZE(scanscache)
    if idx < 0 or idx >= nscans:
        PyErr_Format(PyExc_IndexError, "Scan group index out of bound: %d.", idx)
    _cache = <ScanND>PyList_GET_ITEM(scanscache, idx)
    if _cache is not None:
        return _cache
    scan = copy_scannd(<ScanND>PyList_GET_ITEM(self.scans, idx))
    base = getfullscan(self, scan.baseidx, True)
    cdef getfullscan_cbs_data data
    data.scan = <PyObject*>scan
    # Merge the fixed parameters
    foreach_nondict(getfullscan_param_cb, &data, base.fixed)
    # Merge the variable parameters
    cdef int scanid
    basevars = base.vars
    for scanid in range(PyList_GET_SIZE(basevars)):
        data.idxsz = scanid
        foreach_nondict(getfullscan_var_cb, &data,
                        (<Scan1D>PyList_GET_ITEM(basevars, scanid)).params)

    vars = scan.vars
    for scanid in range(PyList_GET_SIZE(vars)):
        data.idxsz = 0
        s1d = <Scan1D>PyList_GET_ITEM(vars, scanid)
        foreach_nondict(getfullscan_count_vars_cb, &data, s1d.params)
        s1d.size = data.idxsz
    scanscache[idx] = scan
    return scan
