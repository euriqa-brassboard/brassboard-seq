# cython: language_level=3

from brassboard_seq cimport rtval

def new_invalid_rtval():
    # This should only happen if something really wrong happens.
    # We'll just test that we behave reasonably enough.
    # (it's unavoidable that we'll crash in some cases)
    rv = rtval._new_rtval(<rtval.ValueType>1000)
    rv.arg0 = rtval.new_const(1)
    rv.arg1 = rtval.new_const(1)
    return rv
