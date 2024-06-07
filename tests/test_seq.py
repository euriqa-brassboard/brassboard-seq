#

from brassboard_seq.config import Config
from brassboard_seq import seq
import pytest

def test_seq():
    with pytest.raises(TypeError):
        seq.SeqInfo()
    with pytest.raises(TypeError):
        seq.TimeSeq()
    with pytest.raises(TypeError):
        seq.TimeStep()
    with pytest.raises(TypeError):
        seq.SubSeq()

    conf = Config()
    s = seq.Seq(conf)
    assert s.start_time is None
    assert str(s.end_time) == '0 ps'
