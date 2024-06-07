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
    with pytest.raises(TypeError):
        seq.Seq()

    conf = Config()
    s = seq.new_seq(conf)
    assert s.start_time is None
    assert str(s.end_time) == '0.0 ps'
