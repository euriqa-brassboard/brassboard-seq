#

from brassboard_seq.config import Config
import pytest

def test_config():
    conf = Config()
    with pytest.raises(ValueError, match='^Unsupported channel'):
        conf.translate_channel('artiq/b')
    with pytest.raises(TypeError, match="^Unexpected type '.*' for prefix"):
        conf.add_supported_prefix([])
    conf.add_supported_prefix('artiq')
    assert conf.translate_channel('artiq/b') == ('artiq', 'b')
    with pytest.raises(TypeError, match="^Unexpected type '.*' for name"):
        conf.add_channel_alias(34, "abc")
    with pytest.raises(TypeError, match="^Unexpected type '.*' for target"):
        conf.add_channel_alias("abc", ())
    with pytest.raises(ValueError, match='^Channel alias name may not contain'):
        conf.add_channel_alias('a/b', 'c/d')

    conf.add_channel_alias('abc', 'artiq/cc')
    with pytest.raises(TypeError, match="^Unexpected type '.*' for name"):
        conf.translate_channel(22)
    assert conf.translate_channel('abc/dd') == ('artiq', 'cc', 'dd')
    assert conf.translate_channel('abc/dd') == ('artiq', 'cc', 'dd')
    assert conf.translate_channel('abc') == ('artiq', 'cc')
    assert conf.translate_channel('abc') == ('artiq', 'cc')

    conf.add_channel_alias('a', 'a/b')
    with pytest.raises(ValueError, match='^Channel alias loop detected'):
        conf.translate_channel('a/b')
