#

from brassboard_seq import action
import brassboard_seq_test_utils as test_utils
import pytest

def check_action_str(action, s):
    assert str(action) == s
    assert repr(action) == s

def test_action():
    with pytest.raises(TypeError):
        action.Action()
    a1 = test_utils.new_action(1.2, True, True, False, {}, 0)
    test_utils.action_set_tid(a1, 1)
    assert test_utils.action_get_aid(a1) == 0
    a2 = test_utils.new_action(1.2, True, False, False, dict(a=1, b=2), 3)
    assert test_utils.action_get_aid(a2) == 3
    test_utils.action_set_tid(a2, 3)
    a3 = test_utils.new_action(2.3, False, True, True, {}, 2)
    assert test_utils.action_get_aid(a3) == 2
    test_utils.action_set_tid(a3, 2)
    check_action_str(a1, "Pulse(1.2)")
    check_action_str(a2, "Set(1.2, a=1, b=2)")
    check_action_str(a3, "Pulse(2.3, cond=False, exact_time=True)")

    assert a1 < a2
    assert a2 > a1
    assert a1 != a2
    assert not (a1 == a2)
    assert a1 == a1

    assert a1 < a3
    assert a3 > a1

    assert a2 > a3
    assert a3 < a2

    with pytest.raises(TypeError):
        a1 <= a2
    with pytest.raises(TypeError):
        a2 >= a1
    with pytest.raises(TypeError):
        a2 >= 1

    assert (a2 == 1) is False
    assert (a2 != 0) is True
