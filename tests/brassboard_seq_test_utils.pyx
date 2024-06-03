# cython: language_level=3

from brassboard_seq cimport action

def new_action(value, cond, bint is_pulse, bint exact_time, dict kws):
    return action.new_action(value, cond, is_pulse, exact_time, kws)

def action_set_tid(action.Action action, int tid):
    action.tid = tid
