#

from brassboard_seq.seq import Seq
from brassboard_seq.config import Config

conf = Config()
conf.add_supported_prefix('artiq')

def test(n):
    s = Seq(conf)
    ch1 = s.get_channel_id('artiq/ttl1')
    ch2 = s.get_channel_id('artiq/ttl2')

    for i in range(n):
        s.add_step(1).set(ch1, True)
        s.add_step(1).pulse(ch2, True)

    return s

def test2(n, m):
    for i in range(m):
        test(n)

test2(10000, 1000)
