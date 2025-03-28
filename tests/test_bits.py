#

import py_test_utils as test_utils

import pytest
import random
import itertools

def rand_bits(cls):
    n, bits, sign = cls.spec()
    maxele = (1 << bits) - 1
    v0 = random.randint(0, maxele)
    if sign and (v0 << 1) > maxele:
        self = cls(v0 - (1 << bits))
        assert self[0] == v0 - (1 << bits)
    else:
        self = cls(v0)
        assert self[0] == v0
    vi = v0
    for i in range(1, n):
        v = random.randint(0, maxele)
        vi |= v << (i * bits)
        if sign and (v << 1) > maxele:
            self[i] = v - (1 << bits)
            assert self[i] == v - (1 << bits)
        else:
            self[i] = v
            assert self[i] == v
    assert int(self) == vi
    rs = repr(self)
    assert int(rs, 0) == vi
    assert self.bytes() == vi.to_bytes(n * bits // 8, 'little')
    assert rs == '0x' + str(self)
    assert int(test_utils.Bits_i32x5(self)) == vi & ((1 << 160) - 1)
    assert int(test_utils.Bits_i64x4(self)) == vi & ((1 << 256) - 1)
    assert int(test_utils.Bits_u64x4(self)) == vi & ((1 << 256) - 1)
    assert int(test_utils.Bits_i8x43(self)) == vi & ((1 << 344) - 1)
    assert self == cls(self)
    assert not (self != cls(self))
    return self

def check_unary(self):
    n, bits, sign = self.spec()
    total_bits = n * bits
    mask = (1 << total_bits) - 1
    assert int(self << total_bits) == 0
    assert int(self >> total_bits) == 0
    assert int(self << -total_bits) == 0
    assert int(self >> -total_bits) == 0
    vi = int(self)
    assert int(~self) == mask ^ vi
    for i in range(total_bits):
        vl = int(self << i)
        vr = int(self >> i)
        assert int(self << -i) == vr
        assert int(self >> -i) == vl
        assert vr == vi >> i
        assert vl == (vi << i) & mask

def check_binary(b1, b2):
    n1, bits1, sign1 = b1.spec()
    n2, bits2, sign2 = b2.spec()
    total_bits1 = n1 * bits1
    total_bits2 = n2 * bits2
    cls1 = type(b1)
    cls2 = type(b2)
    cls = cls1 if total_bits1 >= total_bits2 else cls2
    v1 = int(b1)
    v2 = int(b2)

    bor = b1 | b2
    band = b1 & b2
    bxor = b1 ^ b2

    vor = v1 | v2
    vand = v1 & v2
    vxor = v1 ^ v2

    assert type(bor) is cls
    assert type(band) is cls
    assert type(bxor) is cls

    assert int(bor) == vor
    assert int(band) == vand
    assert int(bxor) == vxor

    bor1 = cls1(b1)
    bor1 |= b2
    band1 = cls1(b1)
    band1 &= b2
    bxor1 = cls1(b1)
    bxor1 ^= b2

    assert int(bor1) == vor & ((1 << total_bits1) - 1)
    assert int(band1) == vand & ((1 << total_bits1) - 1)
    assert int(bxor1) == vxor & ((1 << total_bits1) - 1)

    bor2 = cls2(b2)
    bor2 |= b1
    band2 = cls2(b2)
    band2 &= b1
    bxor2 = cls2(b2)
    bxor2 ^= b1

    assert int(bor2) == vor & ((1 << total_bits2) - 1)
    assert int(band2) == vand & ((1 << total_bits2) - 1)
    assert int(bxor2) == vxor & ((1 << total_bits2) - 1)

    if cls1 is cls2:
        assert (b1 == b2) == (v1 == v2)
        assert (b1 != b2) == (v1 != v2)
        assert (b1 < b2) == (v1 < v2)
        assert (b1 > b2) == (v1 > v2)
        assert (b1 <= b2) == (v1 <= v2)
        assert (b1 >= b2) == (v1 >= v2)
    else:
        _b2 = cls1(b2)
        _v2 = v2 & ((1 << total_bits1) - 1)
        assert (b1 == _b2) == (v1 == _v2)
        assert (b1 != _b2) == (v1 != _v2)
        assert (b1 < _b2) == (v1 < _v2)
        assert (b1 > _b2) == (v1 > _v2)
        assert (b1 <= _b2) == (v1 <= _v2)
        assert (b1 >= _b2) == (v1 >= _v2)

        _b1 = cls2(b1)
        _v1 = v1 & ((1 << total_bits2) - 1)
        assert (_b1 == b2) == (_v1 == v2)
        assert (_b1 != b2) == (_v1 != v2)
        assert (_b1 < b2) == (_v1 < v2)
        assert (_b1 > b2) == (_v1 > v2)
        assert (_b1 <= b2) == (_v1 <= v2)
        assert (_b1 >= b2) == (_v1 >= v2)

classes = [test_utils.Bits_i32x5, test_utils.Bits_i64x4,
           test_utils.Bits_u64x4, test_utils.Bits_i8x43]

@pytest.mark.parametrize('cls', classes)
def test_unary(cls):
    n, bits, sign = cls.spec()
    assert not cls()
    assert cls(1)
    total_bits = n * bits
    for b1 in range(total_bits):
        for b2 in range(total_bits):
            mask = cls.get_mask(b1, b2)
            if b1 > b2:
                assert int(mask) == 0
                assert not mask
            else:
                assert int(mask) == ((1 << (b2 - b1 + 1)) - 1) << b1
                assert mask
    for _ in range(1500):
        check_unary(rand_bits(cls))

@pytest.mark.parametrize('cls1,cls2', itertools.product(classes, classes))
def test_binary(cls1, cls2):
    for _ in range(2000):
        check_binary(rand_bits(cls1), rand_bits(cls2))
