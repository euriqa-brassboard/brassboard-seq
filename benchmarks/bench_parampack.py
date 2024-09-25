#

from brassboard_seq.scan import ParamPack

def test():
    ParamPack()
    # ParamPack(a=2, b=3)
    # p = ParamPack(a=2, b=3)
    # p.a(3)
    # p2 = p.c(c=2)
    # p2.c(10)
    # p.b()

for i in range(10_000_000):
    test()
