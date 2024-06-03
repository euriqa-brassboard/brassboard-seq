# cython: language_level=3

from brassboard_seq cimport action, rtval

def new_invalid_rtval():
    # This should only happen if something really wrong happens.
    # We'll just test that we behave reasonably enough.
    # (it's unavoidable that we'll crash in some cases)
    rv = rtval._new_rtval(<rtval.ValueType>1000)
    rv.arg0 = rtval.new_const(1)
    rv.arg1 = rtval.new_const(1)
    return rv

def new_action(value, cond, bint is_pulse, bint exact_time, dict kws, int aid):
    return action.new_action(value, cond, is_pulse, exact_time, kws, aid)

def action_set_tid(action.Action action, int tid):
    action.tid = tid

def action_get_aid(action.Action action):
    return action.aid
