#

from brassboard_seq.config import Config
from brassboard_seq import backend, seq
import pytest

def test_compiler_error():
    s = seq.Seq(Config())
    comp = backend.SeqCompiler(s)
    comp.add_backend('artiq', backend.Backend())
    with pytest.raises(ValueError, match="Backend artiq already exist"):
        comp.add_backend('artiq', backend.Backend())

    conf = Config()
    conf.add_supported_prefix('artiq')
    s = seq.Seq(conf)
    comp = backend.SeqCompiler(s)
    assert s.get_channel_id('artiq/a') == 0
    with pytest.raises(ValueError, match="Unhandled channel: artiq/a"):
        comp.finalize()
