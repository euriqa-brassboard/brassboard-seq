#

from brassboard_seq import event_time, rtval
import brassboard_seq_test_utils as test_utils

import pytest

def test_event_time_constructors():
    with pytest.raises(TypeError):
        event_time.TimeManager()
    with pytest.raises(TypeError):
        event_time.EventTime()
    with pytest.raises(TypeError):
        event_time.EventTimeDiff()

def test_round_time():
    assert event_time.time_scale() == 1000000000000
    assert test_utils.round_time(1.2) == 1200_000_000_000
    v = rtval.new_extern(lambda: 2.3)
    s = str(v)
    assert str(test_utils.round_time(v)) == f'int64({s} * 1000000000000)'

def test_event_time1():
    mgr = test_utils.new_time_manager()
    t0 = test_utils.time_manager_new_time(mgr, None, 0, False, True, None)
    assert str(t0) == '0 ps'
    assert repr(t0) == str(t0)
    assert test_utils.event_time_id(t0) == 0
    assert test_utils.event_time_get_static(t0) == -1

    t1 = test_utils.time_manager_new_time(mgr, t0, 1000, False, True, None)
    with pytest.raises(ValueError, match="Cannot modify non-floating time"):
        test_utils.event_time_set_base(t1, t0, 3400)
    assert str(t1) == 'T[0] + 1 ns'
    assert test_utils.event_time_id(t1) == 1
    assert test_utils.event_time_get_static(t1) == -1

    with pytest.raises(ValueError, match="Time delay cannot be negative"):
        test_utils.time_manager_new_time(mgr, t0, -1000, False, True, None)

    t2 = test_utils.time_manager_new_time(mgr, t1, 100, False, True, t0)
    assert str(t2) == 'T[1]; wait_for(T[0] + 100 ps)'
    assert test_utils.event_time_id(t2) == 2
    assert test_utils.event_time_get_static(t2) == -1

    t3 = test_utils.time_manager_new_time(mgr, t1, 2000, False, True, t0)
    assert str(t3) == 'T[1]; wait_for(T[0] + 2 ns)'
    assert test_utils.event_time_id(t3) == 3
    assert test_utils.event_time_get_static(t3) == -1

    t4 = test_utils.time_manager_new_time(mgr, t2, 2000, False, False, t1)
    assert str(t4) == 'T[2]; wait_for(T[1] + 2 ns; if False)'
    assert test_utils.event_time_id(t4) == 4
    assert test_utils.event_time_get_static(t4) == -1

    t5 = test_utils.time_manager_new_round_time(mgr, t4, 3e-9, False, False, None)
    assert str(t5) == 'T[4] + (3 ns; if False)'
    assert test_utils.event_time_id(t5) == 5
    assert test_utils.event_time_get_static(t5) == -1

    t6 = test_utils.time_manager_new_time(mgr, None, 0, True, True, None)
    assert str(t6) == '<floating>'
    assert test_utils.event_time_id(t6) == 6
    assert test_utils.event_time_get_static(t6) == -1
    with pytest.raises(ValueError, match="Time delay cannot be negative"):
        test_utils.event_time_set_base(t6, t5, -3400)
    test_utils.event_time_set_base(t6, t5, 3400_000)
    assert str(t6) == 'T[5] + 3.4 us'
    with pytest.raises(ValueError, match="Cannot modify non-floating time"):
        test_utils.event_time_set_base(t6, t5, 3400)

    with pytest.raises(RuntimeError, match="Event times not finalized"):
        test_utils.event_time_is_ordered(t0, t1)

    with pytest.raises(RuntimeError, match="Event times not finalized"):
        test_utils.time_manager_compute_all_times(mgr, 1)

    test_utils.time_manager_finalize(mgr)
    assert test_utils.time_manager_nchain(mgr) == 2
    assert test_utils.event_time_get_static(t0) == 0
    assert test_utils.event_time_get_static(t1) == 1000
    assert test_utils.event_time_get_static(t2) == 1000
    assert test_utils.event_time_get_static(t3) == 2000
    assert test_utils.event_time_get_static(t4) == 1000
    assert test_utils.event_time_get_static(t5) == 1000
    assert test_utils.event_time_get_static(t6) == 3401_000

    with pytest.raises(RuntimeError, match="Cannot allocate more time"):
        test_utils.time_manager_new_time(mgr, None, 0, True, True, None)
    with pytest.raises(RuntimeError, match="Event times already"):
        test_utils.time_manager_finalize(mgr)

    assert test_utils.event_time_id(t0) == 0
    assert test_utils.event_time_id(t1) == 1
    assert test_utils.event_time_id(t2) == 2
    assert test_utils.event_time_id(t3) == 3
    assert test_utils.event_time_id(t4) == 4
    assert test_utils.event_time_id(t5) == 5
    assert test_utils.event_time_id(t6) == 6

    assert test_utils.event_time_is_ordered(t0, t0) == 'OrderEqual'
    assert test_utils.event_time_is_ordered(t0, t1) == 'OrderBefore'
    assert test_utils.event_time_is_ordered(t1, t0) == 'OrderAfter'

    assert test_utils.event_time_is_ordered(t1, t2) == 'OrderBefore'
    assert test_utils.event_time_is_ordered(t2, t1) == 'OrderAfter'

    assert test_utils.event_time_is_ordered(t1, t3) == 'OrderBefore'
    assert test_utils.event_time_is_ordered(t2, t3) == 'NoOrder'

    assert test_utils.event_time_is_ordered(t1, t4) == 'OrderBefore'
    assert test_utils.event_time_is_ordered(t2, t4) == 'OrderBefore'
    assert test_utils.event_time_is_ordered(t3, t4) == 'NoOrder'

    assert test_utils.event_time_is_ordered(t4, t5) == 'OrderBefore'
    assert test_utils.event_time_is_ordered(t3, t5) == 'NoOrder'

    assert test_utils.event_time_is_ordered(t5, t6) == 'OrderBefore'
    assert test_utils.event_time_is_ordered(t3, t6) == 'NoOrder'

    max_time, time_values = test_utils.time_manager_compute_all_times(mgr, 1)
    assert time_values == [0, 1000, 1000, 2000, 1000, 1000, 3401_000]
    assert max_time == 3401_000

    mgr = test_utils.new_time_manager()
    t0 = test_utils.time_manager_new_time(mgr, None, 0, False, True, None)

    with pytest.raises(ValueError, match="Cannot take the difference"):
        t0 - t1

    t1 = test_utils.time_manager_new_time(mgr, None, 0, True, True, None)
    t2 = test_utils.time_manager_new_time(mgr, None, 0, True, True, None)

    test_utils.event_time_set_base(t1, t2, 10)
    test_utils.event_time_set_base(t2, t1, 10)

    with pytest.raises(ValueError, match="Time loop"):
        test_utils.time_manager_finalize(mgr)
    with pytest.raises(RuntimeError, match="Event times already"):
        test_utils.time_manager_finalize(mgr)

    with pytest.raises(RuntimeError, match="Cannot allocate more time"):
        # A failed finalize would also freeze the time manager
        test_utils.time_manager_new_time(mgr, None, 0, True, True, None)

def test_event_time1_rt():
    mgr = test_utils.new_time_manager()
    t0 = test_utils.time_manager_new_time(mgr, None, 0, False, True, None)
    assert str(t0) == '0 ps'
    assert repr(t0) == str(t0)
    assert test_utils.event_time_id(t0) == 0
    assert test_utils.event_time_get_static(t0) == -1

    v1 = rtval.new_extern(lambda: 1000)
    t1 = test_utils.time_manager_new_time(mgr, t0, v1, False, True, None)
    with pytest.raises(ValueError, match="Cannot modify non-floating time"):
        test_utils.event_time_set_base(t1, t0, 3400)
    assert str(t1) == f'T[0] + {v1}'
    assert test_utils.event_time_id(t1) == 1
    assert test_utils.event_time_get_static(t1) == -1

    t2 = test_utils.time_manager_new_time(mgr, t1, 100, False, True, t0)
    assert str(t2) == 'T[1]; wait_for(T[0] + 100 ps)'
    assert test_utils.event_time_id(t2) == 2
    assert test_utils.event_time_get_static(t2) == -1

    t3 = test_utils.time_manager_new_time(mgr, t1, 2000, False, True, t0)
    assert str(t3) == 'T[1]; wait_for(T[0] + 2 ns)'
    assert test_utils.event_time_id(t3) == 3
    assert test_utils.event_time_get_static(t3) == -1

    t4 = test_utils.time_manager_new_time(mgr, t2, 2000, False, False, t1)
    assert str(t4) == 'T[2]; wait_for(T[1] + 2 ns; if False)'
    assert test_utils.event_time_id(t4) == 4
    assert test_utils.event_time_get_static(t4) == -1

    t5 = test_utils.time_manager_new_round_time(mgr, t4, 3e-9, False, False, None)
    assert str(t5) == 'T[4] + (3 ns; if False)'
    assert test_utils.event_time_id(t5) == 5
    assert test_utils.event_time_get_static(t5) == -1

    t6 = test_utils.time_manager_new_time(mgr, None, 0, True, True, None)
    assert str(t6) == '<floating>'
    assert test_utils.event_time_id(t6) == 6
    assert test_utils.event_time_get_static(t6) == -1
    test_utils.event_time_set_base(t6, t5, 3400_000)
    assert str(t6) == 'T[5] + 3.4 us'
    with pytest.raises(ValueError, match="Cannot modify non-floating time"):
        test_utils.event_time_set_base(t6, t5, 3400)

    with pytest.raises(RuntimeError, match="Event times not finalized"):
        test_utils.event_time_is_ordered(t0, t1)

    with pytest.raises(RuntimeError, match="Event times not finalized"):
        test_utils.time_manager_compute_all_times(mgr, 1)

    test_utils.time_manager_finalize(mgr)
    assert test_utils.time_manager_nchain(mgr) == 2
    assert test_utils.event_time_get_static(t0) == 0
    assert test_utils.event_time_get_static(t1) == -1
    assert test_utils.event_time_get_static(t2) == -1
    assert test_utils.event_time_get_static(t3) == -1
    assert test_utils.event_time_get_static(t4) == -1
    assert test_utils.event_time_get_static(t5) == -1
    assert test_utils.event_time_get_static(t6) == -1

    assert test_utils.event_time_id(t0) == 0
    assert test_utils.event_time_id(t1) == 1
    assert test_utils.event_time_id(t2) == 2
    assert test_utils.event_time_id(t3) == 3
    assert test_utils.event_time_id(t4) == 4
    assert test_utils.event_time_id(t5) == 5
    assert test_utils.event_time_id(t6) == 6

    assert test_utils.event_time_is_ordered(t0, t0) == 'OrderEqual'
    assert test_utils.event_time_is_ordered(t0, t1) == 'OrderBefore'
    assert test_utils.event_time_is_ordered(t1, t0) == 'OrderAfter'

    assert test_utils.event_time_is_ordered(t1, t2) == 'OrderBefore'
    assert test_utils.event_time_is_ordered(t2, t1) == 'OrderAfter'

    assert test_utils.event_time_is_ordered(t1, t3) == 'OrderBefore'
    assert test_utils.event_time_is_ordered(t2, t3) == 'NoOrder'

    assert test_utils.event_time_is_ordered(t1, t4) == 'OrderBefore'
    assert test_utils.event_time_is_ordered(t2, t4) == 'OrderBefore'
    assert test_utils.event_time_is_ordered(t3, t4) == 'NoOrder'

    assert test_utils.event_time_is_ordered(t4, t5) == 'OrderBefore'
    assert test_utils.event_time_is_ordered(t3, t5) == 'NoOrder'

    assert test_utils.event_time_is_ordered(t5, t6) == 'OrderBefore'
    assert test_utils.event_time_is_ordered(t3, t6) == 'NoOrder'

    max_time, time_values = test_utils.time_manager_compute_all_times(mgr, 1)
    assert time_values == [0, 1000, 1000, 2000, 1000, 1000, 3401_000]
    assert max_time == 3401_000

def test_event_time_static0():
    mgr = test_utils.new_time_manager()
    t0 = test_utils.time_manager_new_time(mgr, None, 0, False, True, None)
    assert str(t0) == '0 ps'
    assert repr(t0) == str(t0)
    assert test_utils.event_time_id(t0) == 0
    assert test_utils.event_time_get_static(t0) == -1

    c = rtval.new_extern(lambda: True)
    t1 = test_utils.time_manager_new_time(mgr, t0, 0, False, c, None)
    assert str(t1) == f'T[0] + (0 ps; if {c})'
    assert repr(t1) == str(t1)
    assert test_utils.event_time_id(t1) == 1
    assert test_utils.event_time_get_static(t1) == -1

    test_utils.time_manager_finalize(mgr)
    assert test_utils.time_manager_nchain(mgr) == 1
    assert test_utils.event_time_get_static(t0) == 0
    assert test_utils.event_time_get_static(t1) == 0

    assert test_utils.event_time_is_ordered(t0, t0) == 'OrderEqual'
    assert test_utils.event_time_is_ordered(t0, t1) == 'OrderBefore'
    assert test_utils.event_time_is_ordered(t1, t0) == 'OrderAfter'

    max_time, time_values = test_utils.time_manager_compute_all_times(mgr, 1)
    assert time_values == [0, 0]
    assert max_time == 0

    mgr = test_utils.new_time_manager()
    t0 = test_utils.time_manager_new_time(mgr, None, 0, False, True, None)
    assert str(t0) == '0 ps'
    assert repr(t0) == str(t0)
    assert test_utils.event_time_id(t0) == 0
    assert test_utils.event_time_get_static(t0) == -1

    c = rtval.new_extern(lambda: True)
    t1 = test_utils.time_manager_new_time(mgr, t0, 0, False, c, t0)
    assert str(t1) == f'T[0]; wait_for(T[0] + 0 ps; if {c})'
    assert repr(t1) == str(t1)
    assert test_utils.event_time_id(t1) == 1
    assert test_utils.event_time_get_static(t1) == -1

    test_utils.time_manager_finalize(mgr)
    assert test_utils.time_manager_nchain(mgr) == 1
    assert test_utils.event_time_get_static(t0) == 0
    assert test_utils.event_time_get_static(t1) == -1

    assert test_utils.event_time_is_ordered(t0, t0) == 'OrderEqual'
    assert test_utils.event_time_is_ordered(t0, t1) == 'OrderBefore'
    assert test_utils.event_time_is_ordered(t1, t0) == 'OrderAfter'

    max_time, time_values = test_utils.time_manager_compute_all_times(mgr, 1)
    assert time_values == [0, 0]
    assert max_time == 0

def test_event_time2():
    mgr = test_utils.new_time_manager()
    t1 = test_utils.time_manager_new_time(mgr, None, 0, True, True, None)

    with pytest.raises(RuntimeError, match="Event time still floating"):
        test_utils.time_manager_finalize(mgr)
    with pytest.raises(RuntimeError, match="Event times already"):
        test_utils.time_manager_finalize(mgr)

def test_event_time3():
    mgr = test_utils.new_time_manager()
    t0 = test_utils.time_manager_new_time(mgr, None, 0, False, True, None)
    assert str(t0) == '0 ps'
    assert repr(t0) == str(t0)
    assert test_utils.event_time_id(t0) == 0

    t1 = test_utils.time_manager_new_round_time(mgr, t0, 0.01,
                                                False, True, None)
    assert str(t1) == 'T[0] + 10 ms'
    assert test_utils.event_time_id(t1) == 1

    t2 = test_utils.time_manager_new_time(mgr, t1, test_utils.round_time(1.2),
                                          False, True, None)
    assert str(t2) == 'T[1] + 1.2 s'
    assert test_utils.event_time_id(t2) == 2

    actual_value = -1.0
    v = rtval.new_extern(lambda: actual_value)
    s = str(v)
    t3 = test_utils.time_manager_new_round_time(mgr, t1, v, False, True, t2)
    with pytest.raises(ValueError, match="Cannot modify non-floating time"):
        test_utils.event_time_set_base(t3, t1, v)
    assert str(t3) == f'T[1]; wait_for(T[2] + int64({s} * 1000000000000))'
    assert test_utils.event_time_id(t3) == 3

    test_utils.time_manager_finalize(mgr)
    with pytest.raises(RuntimeError, match="Cannot allocate more time"):
        test_utils.time_manager_new_time(mgr, t1, test_utils.round_time(v),
                                         False, True, t2)

    assert test_utils.time_manager_nchain(mgr) == 1

    assert test_utils.event_time_id(t0) == 0
    assert test_utils.event_time_id(t1) == 1
    assert test_utils.event_time_id(t2) == 2
    assert test_utils.event_time_id(t3) == 3

    assert test_utils.event_time_get_static(t0) == 0
    assert test_utils.event_time_get_static(t1) == 10_000_000_000
    assert test_utils.event_time_get_static(t2) == 1_210_000_000_000
    assert test_utils.event_time_get_static(t3) == -1

    assert test_utils.event_time_is_ordered(t0, t1) == 'OrderBefore'
    assert test_utils.event_time_is_ordered(t1, t2) == 'OrderBefore'
    assert test_utils.event_time_is_ordered(t2, t3) == 'OrderBefore'

    with pytest.raises(ValueError, match="Time delay cannot be negative"):
        test_utils.time_manager_compute_all_times(mgr, 0)

    actual_value = 1.0

    max_time, time_values = test_utils.time_manager_compute_all_times(mgr, 1)
    assert time_values == [0, 10_000_000_000, 1_210_000_000_000, 2_210_000_000_000]
    assert max_time == 2_210_000_000_000

    actual_value = 2.0

    max_time, time_values = test_utils.time_manager_compute_all_times(mgr, 2)
    assert time_values == [0, 10_000_000_000, 1_210_000_000_000, 3_210_000_000_000]
    assert max_time == 3_210_000_000_000

def test_event_time3_rt():
    mgr = test_utils.new_time_manager()
    t0 = test_utils.time_manager_new_time(mgr, None, 0, False, True, None)
    assert str(t0) == '0 ps'
    assert repr(t0) == str(t0)
    assert test_utils.event_time_id(t0) == 0

    v0 = rtval.new_extern(lambda: 0.01)
    t1 = test_utils.time_manager_new_round_time(mgr, t0, v0,
                                                False, True, None)
    assert str(t1) == f'T[0] + int64({v0} * 1000000000000)'
    assert test_utils.event_time_id(t1) == 1

    t2 = test_utils.time_manager_new_time(mgr, t1, test_utils.round_time(1.2),
                                          False, True, None)
    assert str(t2) == 'T[1] + 1.2 s'
    assert test_utils.event_time_id(t2) == 2

    actual_value = -1.0
    v = rtval.new_extern(lambda: actual_value)
    s = str(v)
    t3 = test_utils.time_manager_new_round_time(mgr, t1, v, False, True, t2)
    with pytest.raises(ValueError, match="Cannot modify non-floating time"):
        test_utils.event_time_set_base(t3, t1, v)
    assert str(t3) == f'T[1]; wait_for(T[2] + int64({s} * 1000000000000))'
    assert test_utils.event_time_id(t3) == 3

    test_utils.time_manager_finalize(mgr)
    with pytest.raises(RuntimeError, match="Cannot allocate more time"):
        test_utils.time_manager_new_time(mgr, t1, test_utils.round_time(v),
                                         False, True, t2)

    assert test_utils.time_manager_nchain(mgr) == 1

    assert test_utils.event_time_id(t0) == 0
    assert test_utils.event_time_id(t1) == 1
    assert test_utils.event_time_id(t2) == 2
    assert test_utils.event_time_id(t3) == 3

    assert test_utils.event_time_get_static(t0) == 0
    assert test_utils.event_time_get_static(t1) == -1
    assert test_utils.event_time_get_static(t2) == -1
    assert test_utils.event_time_get_static(t3) == -1

    assert test_utils.event_time_is_ordered(t0, t1) == 'OrderBefore'
    assert test_utils.event_time_is_ordered(t1, t2) == 'OrderBefore'
    assert test_utils.event_time_is_ordered(t2, t3) == 'OrderBefore'

    with pytest.raises(ValueError, match="Time delay cannot be negative"):
        test_utils.time_manager_compute_all_times(mgr, 0)

    actual_value = 1.0

    max_time, time_values = test_utils.time_manager_compute_all_times(mgr, 1)
    assert time_values == [0, 10_000_000_000, 1_210_000_000_000, 2_210_000_000_000]
    assert max_time == 2_210_000_000_000

    actual_value = 2.0

    max_time, time_values = test_utils.time_manager_compute_all_times(mgr, 2)
    assert time_values == [0, 10_000_000_000, 1_210_000_000_000, 3_210_000_000_000]
    assert max_time == 3_210_000_000_000

def test_event_time4():
    mgr = test_utils.new_time_manager()
    t0 = test_utils.time_manager_new_time(mgr, None, 0, False, True, None)
    assert str(t0) == '0 ps'
    assert repr(t0) == str(t0)
    assert test_utils.event_time_id(t0) == 0

    t1 = test_utils.time_manager_new_time(mgr, t0, test_utils.round_time(0.01),
                                          False, True, None)
    assert str(t1) == 'T[0] + 10 ms'
    assert test_utils.event_time_id(t1) == 1

    t2 = test_utils.time_manager_new_time(mgr, t0, test_utils.round_time(0.5),
                                          False, True, None)
    assert str(t2) == 'T[0] + 0.5 s'
    assert test_utils.event_time_id(t2) == 2

    t3 = test_utils.time_manager_new_time(mgr, t1, test_utils.round_time(0.02),
                                          False, True, t2)
    assert str(t3) == f'T[1]; wait_for(T[2] + 20 ms)'
    assert test_utils.event_time_id(t3) == 3

    test_utils.time_manager_finalize(mgr)
    assert test_utils.time_manager_nchain(mgr) == 2

    assert test_utils.event_time_id(t0) == 0
    assert test_utils.event_time_id(t1) == 1
    assert test_utils.event_time_id(t2) == 2
    assert test_utils.event_time_id(t3) == 3

    assert test_utils.event_time_get_static(t0) == 0
    assert test_utils.event_time_get_static(t1) == 10_000_000_000
    assert test_utils.event_time_get_static(t2) == 500_000_000_000
    assert test_utils.event_time_get_static(t3) == 520_000_000_000

    assert test_utils.event_time_is_ordered(t0, t1) == 'OrderBefore'
    assert test_utils.event_time_is_ordered(t0, t2) == 'OrderBefore'
    assert test_utils.event_time_is_ordered(t1, t2) == 'NoOrder'
    assert test_utils.event_time_is_ordered(t1, t3) == 'OrderBefore'
    assert test_utils.event_time_is_ordered(t2, t3) == 'OrderBefore'

    max_time, time_values = test_utils.time_manager_compute_all_times(mgr, 1)
    assert time_values == [0, 10_000_000_000, 500_000_000_000, 520_000_000_000]
    assert max_time == 520_000_000_000

def test_event_time4_rt():
    mgr = test_utils.new_time_manager()
    t0 = test_utils.time_manager_new_time(mgr, None, 0, False, True, None)
    assert str(t0) == '0 ps'
    assert repr(t0) == str(t0)
    assert test_utils.event_time_id(t0) == 0

    t1 = test_utils.time_manager_new_time(mgr, t0, test_utils.round_time(0.01),
                                          False, True, None)
    assert str(t1) == 'T[0] + 10 ms'
    assert test_utils.event_time_id(t1) == 1

    v = rtval.new_extern(lambda: 0.5)
    t2 = test_utils.time_manager_new_time(mgr, t0, test_utils.round_time(v),
                                          False, True, None)
    assert str(t2) == f'T[0] + int64({v} * 1000000000000)'
    assert test_utils.event_time_id(t2) == 2

    t3 = test_utils.time_manager_new_time(mgr, t1, test_utils.round_time(0.02),
                                          False, True, t2)
    assert str(t3) == f'T[1]; wait_for(T[2] + 20 ms)'
    assert test_utils.event_time_id(t3) == 3

    test_utils.time_manager_finalize(mgr)
    assert test_utils.time_manager_nchain(mgr) == 2

    assert test_utils.event_time_id(t0) == 0
    assert test_utils.event_time_id(t1) == 1
    assert test_utils.event_time_id(t2) == 2
    assert test_utils.event_time_id(t3) == 3

    assert test_utils.event_time_get_static(t0) == 0
    assert test_utils.event_time_get_static(t1) == 10_000_000_000
    assert test_utils.event_time_get_static(t2) == -1
    assert test_utils.event_time_get_static(t3) == -1

    assert test_utils.event_time_is_ordered(t0, t1) == 'OrderBefore'
    assert test_utils.event_time_is_ordered(t0, t2) == 'OrderBefore'
    assert test_utils.event_time_is_ordered(t1, t2) == 'NoOrder'
    assert test_utils.event_time_is_ordered(t1, t3) == 'OrderBefore'
    assert test_utils.event_time_is_ordered(t2, t3) == 'OrderBefore'

    max_time, time_values = test_utils.time_manager_compute_all_times(mgr, 1)
    assert time_values == [0, 10_000_000_000, 500_000_000_000, 520_000_000_000]
    assert max_time == 520_000_000_000

def test_event_time5():
    mgr = test_utils.new_time_manager()
    t0 = test_utils.time_manager_new_time(mgr, None, 0, False, True, None)
    assert str(t0) == '0 ps'
    assert repr(t0) == str(t0)
    assert test_utils.event_time_id(t0) == 0

    t1 = test_utils.time_manager_new_time(mgr, t0, test_utils.round_time(0.01),
                                          False, True, None)
    assert str(t1) == 'T[0] + 10 ms'
    assert test_utils.event_time_id(t1) == 1

    t2 = test_utils.time_manager_new_time(mgr, t0, test_utils.round_time(0.5),
                                          False, True, None)
    assert str(t2) == 'T[0] + 0.5 s'
    assert test_utils.event_time_id(t2) == 2

    t3 = test_utils.time_manager_new_time(mgr, t2, test_utils.round_time(0.02),
                                          False, True, t1)
    assert str(t3) == f'T[2]; wait_for(T[1] + 20 ms)'
    assert test_utils.event_time_id(t3) == 3

    test_utils.time_manager_finalize(mgr)
    assert test_utils.time_manager_nchain(mgr) == 2

    assert test_utils.event_time_id(t0) == 0
    assert test_utils.event_time_id(t1) == 1
    assert test_utils.event_time_id(t2) == 2
    assert test_utils.event_time_id(t3) == 3

    assert test_utils.event_time_get_static(t0) == 0
    assert test_utils.event_time_get_static(t1) == 10_000_000_000
    assert test_utils.event_time_get_static(t2) == 500_000_000_000
    assert test_utils.event_time_get_static(t3) == 500_000_000_000

    assert test_utils.event_time_is_ordered(t0, t1) == 'OrderBefore'
    assert test_utils.event_time_is_ordered(t0, t2) == 'OrderBefore'
    assert test_utils.event_time_is_ordered(t1, t2) == 'NoOrder'
    assert test_utils.event_time_is_ordered(t1, t3) == 'OrderBefore'
    assert test_utils.event_time_is_ordered(t2, t3) == 'OrderBefore'

    max_time, time_values = test_utils.time_manager_compute_all_times(mgr, 1)
    assert time_values == [0, 10_000_000_000, 500_000_000_000, 500_000_000_000]
    assert max_time == 500_000_000_000

def test_event_time5_rt():
    mgr = test_utils.new_time_manager()
    t0 = test_utils.time_manager_new_time(mgr, None, 0, False, True, None)
    assert str(t0) == '0 ps'
    assert repr(t0) == str(t0)
    assert test_utils.event_time_id(t0) == 0

    v = rtval.new_extern(lambda: 0.01)
    t1 = test_utils.time_manager_new_time(mgr, t0, test_utils.round_time(v),
                                          False, True, None)
    assert str(t1) == f'T[0] + int64({v} * 1000000000000)'
    assert test_utils.event_time_id(t1) == 1

    t2 = test_utils.time_manager_new_time(mgr, t0, test_utils.round_time(0.5),
                                          False, True, None)
    assert str(t2) == 'T[0] + 0.5 s'
    assert test_utils.event_time_id(t2) == 2

    t3 = test_utils.time_manager_new_time(mgr, t2, test_utils.round_time(0.02),
                                          False, True, t1)
    assert str(t3) == f'T[2]; wait_for(T[1] + 20 ms)'
    assert test_utils.event_time_id(t3) == 3

    test_utils.time_manager_finalize(mgr)
    assert test_utils.time_manager_nchain(mgr) == 2

    assert test_utils.event_time_id(t0) == 0
    assert test_utils.event_time_id(t1) == 1
    assert test_utils.event_time_id(t2) == 2
    assert test_utils.event_time_id(t3) == 3

    assert test_utils.event_time_get_static(t0) == 0
    assert test_utils.event_time_get_static(t1) == -1
    assert test_utils.event_time_get_static(t2) == 500_000_000_000
    assert test_utils.event_time_get_static(t3) == -1

    assert test_utils.event_time_is_ordered(t0, t1) == 'OrderBefore'
    assert test_utils.event_time_is_ordered(t0, t2) == 'OrderBefore'
    assert test_utils.event_time_is_ordered(t1, t2) == 'NoOrder'
    assert test_utils.event_time_is_ordered(t1, t3) == 'OrderBefore'
    assert test_utils.event_time_is_ordered(t2, t3) == 'OrderBefore'

    max_time, time_values = test_utils.time_manager_compute_all_times(mgr, 1)
    assert time_values == [0, 10_000_000_000, 500_000_000_000, 500_000_000_000]
    assert max_time == 500_000_000_000

def test_time_diff():
    mgr = test_utils.new_time_manager()
    t00 = test_utils.time_manager_new_time(mgr, None, 0, False, True, None)
    t01 = test_utils.time_manager_new_time(mgr, None, 0, False, True, None)

    dt = t00 - t01
    assert str(dt) == 'extern_age(T[0] - T[1])'
    with pytest.raises(RuntimeError, match="Event times not finalized"):
        rtval.get_value(dt, 0)

    test_utils.time_manager_finalize(mgr)
    assert test_utils.time_manager_nchain(mgr) == 2

    assert rtval.get_value(dt, 1) == 0.0

def test_time_diff2():
    mgr = test_utils.new_time_manager()
    t0 = test_utils.time_manager_new_time(mgr, None, 0, False, True, None)
    t1 = test_utils.time_manager_new_time(mgr, t0, 10, False, True, None)
    t2 = test_utils.time_manager_new_time(mgr, t0, 1000, False, True, None)

    dt = t2 - t1

    test_utils.time_manager_finalize(mgr)
    assert test_utils.time_manager_nchain(mgr) == 2

    assert rtval.get_value(dt, 1) == 990e-12

def test_time_diff3():
    mgr = test_utils.new_time_manager()
    t0 = test_utils.time_manager_new_time(mgr, None, 0, False, True, None)
    t1 = test_utils.time_manager_new_time(mgr, t0, 10, False, True, None)
    t2 = test_utils.time_manager_new_time(mgr, None, 0, True, True, None)

    dt = t2 - t1
    test_utils.event_time_set_base(t2, t1, test_utils.round_time(dt))
    assert str(t2) == 'T[1] + int64(extern_age(T[2] - T[1]) * 1000000000000)'

    test_utils.time_manager_finalize(mgr)
    assert test_utils.time_manager_nchain(mgr) == 1

    with pytest.raises(ValueError, match="Recursive value dependency"):
        rtval.get_value(dt, 1)

    with pytest.raises(ValueError, match="Recursive value dependency"):
        test_utils.time_manager_compute_all_times(mgr, 2)

def test_time_diff4():
    mgr = test_utils.new_time_manager()
    t0 = test_utils.time_manager_new_time(mgr, None, 0, False, True, None)
    t1 = test_utils.time_manager_new_time(mgr, None, 0, True, True, None)
    t2 = test_utils.time_manager_new_time(mgr, t1, 1000, False, True, None)

    dt = t2 - t1
    test_utils.event_time_set_base(t1, t0, test_utils.round_time(dt))
    assert str(t1) == 'T[0] + int64(extern_age(T[2] - T[1]) * 1000000000000)'

    test_utils.time_manager_finalize(mgr)
    assert test_utils.time_manager_nchain(mgr) == 1

    assert rtval.get_value(dt, 1) == 1e-9

    max_time, time_values = test_utils.time_manager_compute_all_times(mgr, 2)
    assert time_values == [0, 1000, 2000]
    assert max_time == 2000

def test_time_diff5():
    mgr = test_utils.new_time_manager()
    t0 = test_utils.time_manager_new_time(mgr, None, 0, False, True, None)
    t1 = test_utils.time_manager_new_time(mgr, None, 0, True, True, None)
    t2 = test_utils.time_manager_new_time(mgr, t1, 1000, False, True, None)
    t3 = test_utils.time_manager_new_time(mgr, t2, 10, False, False, t0)

    dt = t3 - t1
    test_utils.event_time_set_base(t1, t0, test_utils.round_time(dt))
    assert str(t1) == 'T[0] + int64(extern_age(T[3] - T[1]) * 1000000000000)'

    test_utils.time_manager_finalize(mgr)
    assert test_utils.time_manager_nchain(mgr) == 1

    assert rtval.get_value(dt, 1) == 1e-9

    max_time, time_values = test_utils.time_manager_compute_all_times(mgr, 2)
    assert time_values == [0, 1000, 2000, 2000]
    assert max_time == 2000

def test_time_diff6():
    mgr = test_utils.new_time_manager()
    t0 = test_utils.time_manager_new_time(mgr, None, 0, False, True, None)
    t1 = test_utils.time_manager_new_time(mgr, t0, 1000, False, True, None)
    t2 = test_utils.time_manager_new_time(mgr, t1, 1000, False, True, None)
    t3 = test_utils.time_manager_new_time(mgr, t2, 30000, False, True, t0)

    dt = t3 - t1

    test_utils.time_manager_finalize(mgr)
    assert test_utils.time_manager_nchain(mgr) == 1

    assert rtval.get_value(dt, 1) == 29e-9

    max_time, time_values = test_utils.time_manager_compute_all_times(mgr, 2)
    assert time_values == [0, 1000, 2000, 30000]
    assert max_time == 30000

def test_time_diff7():
    mgr = test_utils.new_time_manager()
    t0 = test_utils.time_manager_new_time(mgr, None, 0, False, True, None)
    t1 = test_utils.time_manager_new_time(mgr, t0, 1000, False, True, None)
    t2 = test_utils.time_manager_new_time(mgr, t1, 1000, False, True, None)

    dt = t2 - t1

    test_utils.time_manager_finalize(mgr)
    assert test_utils.time_manager_nchain(mgr) == 1

    assert rtval.get_value(dt, 1) == 1e-9

    max_time, time_values = test_utils.time_manager_compute_all_times(mgr, 2)
    assert time_values == [0, 1000, 2000]
    assert max_time == 2000
