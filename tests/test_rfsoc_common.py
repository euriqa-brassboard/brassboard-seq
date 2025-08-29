#

from rfsoc_test_utils import bitcast_f64_i64, f64parts, encode_pdq_spline, check_spline_shift

import pytest
import random
import struct

def test_bitcast():
    assert bitcast_f64_i64(0.0) == 0
    assert bitcast_f64_i64(-0.0) == -(1 << 63)
    for i in range(10000):
        f1 = random.random()
        f2 = -f1
        assert struct.pack('@d', f1) == struct.pack('@q', bitcast_f64_i64(f1))
        assert struct.pack('@d', f2) == struct.pack('@q', bitcast_f64_i64(f2))

def test_f64parts():
    assert f64parts(0.0) == (0, 1<<52)
    assert f64parts(-0.0) == (0, -1<<52)
    for i in range(10000):
        f1 = random.random()
        if f1 < 2**-1022:
            continue
        f2 = -f1
        f3 = f1 * 2**-1022 * (1 - 2**-52)
        f4 = -f3

        e1, m1 = f64parts(f1)
        e2, m2 = f64parts(f2)
        assert e1 == e2
        assert m1 == -m2
        assert f1 == m1 * 2**(e1 - 1023 - 52)
        assert f2 == m2 * 2**(e2 - 1023 - 52)

        e3, m3 = f64parts(f3)
        e4, m4 = f64parts(f4)
        assert e3 == 0
        assert e4 == 0
        assert m3 == -m4
        assert f3 == (m3 - (1<<52)) * 2**(1 - 1023 - 52)
        assert f4 == (m4 + (1<<52)) * 2**(1 - 1023 - 52)

def mask_u40(u40, shift, fracbit):
    fracbit -= shift
    if fracbit >= 0:
        return u40
    mask = ~((1 << -fracbit) - 1)
    return u40 & mask

def u40_to_i40(u40):
    assert (u40 >> 40) == 0
    if u40 >> 39:
        return u40 - (1 << 40)
    return u40

def sp_inrange(sp):
    return -0.5 < sp < 0.5 - 2**-41

def pdq_spline(_isp0, sp1, sp2, sp3):
    (isp0, isp1, isp2, isp3), shift = encode_pdq_spline(_isp0, sp1, sp2, sp3)
    assert isp0 == _isp0
    assert 0 <= shift <= 11
    isp1 = u40_to_i40(mask_u40(isp1, shift, 0))
    isp2 = u40_to_i40(mask_u40(isp2, shift * 2, 16))
    isp3 = u40_to_i40(mask_u40(isp3, shift * 3, 32))
    if sp_inrange(sp1) and sp_inrange(sp2) and sp_inrange(sp3):
        check_spline_shift(shift, isp1, isp2, isp3)
        assert isp1 * 2**(-40 - shift) == pytest.approx(sp1, rel=2**-40, abs=2**-40)
        assert isp2 * 2**(-40 - shift * 2) == pytest.approx(sp2, rel=2**-40,
                                                            abs=2**-(40 + min(shift * 2, 16)))
        assert isp3 * 2**(-40 - shift * 3) == pytest.approx(sp3, rel=2**-40,
                                                            abs=2**-(40 + min(shift * 3, 32)))
    else:
        assert shift == 0
    return shift, isp1, isp2, isp3

def rand_sp():
    return (random.random() - 0.5) * 2**-random.randint(0, 30)

def test_pdq_spline():
    assert pdq_spline(100, 0, 0, 0) == (11, 0, 0, 0)
    assert pdq_spline(100, -0.0, -0.0, -0.0) == (11, 0, 0, 0)

    assert pdq_spline(100, 0.5, 0, 0) == (0, 0, 0, 0)
    assert pdq_spline(100, -0.5, 0, 0) == (0, 0, 0, 0)
    assert pdq_spline(100, 0.5 * (1 - 2**-39), 0, 0) == (0, 2**39 - 1, 0, 0)
    assert pdq_spline(100, -0.5 * (1 - 2**-41), 0, 0) == (0, -2**39, 0, 0)
    assert pdq_spline(100, -0.5 * (1 - 2**-40), 0, 0) == (0, -2**39 + 1, 0, 0)
    assert pdq_spline(100, -0.5 * (1 - 2**-39), 0, 0) == (0, -2**39 + 1, 0, 0)

    assert pdq_spline(100, 0.25, 0, 0) == (0, 2**38, 0, 0)
    assert pdq_spline(100, -0.25, 0, 0) == (0, -2**38, 0, 0)
    assert pdq_spline(100, 0.25 * (1 - 2**-40), 0, 0) == (0, 2**38, 0, 0)
    assert pdq_spline(100, 0.25 * (1 - 2**-39), 0, 0) == (0, 2**38, 0, 0)
    assert pdq_spline(100, 0.25 * (1 - 2**-38), 0, 0) == (1, 2**39 - 2, 0, 0)
    assert pdq_spline(100, -0.25 * (1 - 2**-40), 0, 0) == (1, -2**39, 0, 0)
    assert pdq_spline(100, -0.25 * (1 - 2**-39), 0, 0) == (1, -2**39 + 2, 0, 0)
    assert pdq_spline(100, -0.25 * (1 - 2**-39), 0, 0) == (1, -2**39 + 2, 0, 0)

    assert pdq_spline(100, 0, 0.125, 0) == (0, 0, 2**37, 0)
    assert pdq_spline(100, 0, -0.125, 0) == (0, 0, -2**37, 0)
    assert pdq_spline(100, 0, 0.125 * (1 - 2**-41), 0) == (0, 0, 2**37, 0)
    assert pdq_spline(100, 0, 0.125 * (1 - 2**-40), 0) == (0, 0, 2**37, 0)
    assert pdq_spline(100, 0, 0.125 * (1 - 2**-39), 0) == (1, 0, 2**39 - 1, 0)
    assert pdq_spline(100, 0, -0.125 * (1 - 2**-41), 0) == (1, 0, -2**39, 0)
    assert pdq_spline(100, 0, -0.125 * (1 - 2**-40), 0) == (1, 0, -2**39 + 1, 0)
    assert pdq_spline(100, 0, -0.125 * (1 - 2**-39), 0) == (1, 0, -2**39 + 1, 0)

    for i in range(10000):
        pdq_spline(random.randint(0, 10000000), rand_sp(), 0, 0)
        pdq_spline(random.randint(0, 10000000), 0, rand_sp(), 0)
        pdq_spline(random.randint(0, 10000000), 0, 0, rand_sp())
        pdq_spline(random.randint(0, 10000000), rand_sp(),
                   rand_sp(), 0)
        pdq_spline(random.randint(0, 10000000), 0, rand_sp(),
                   rand_sp())
        pdq_spline(random.randint(0, 10000000), rand_sp(), 0,
                   rand_sp())
        pdq_spline(random.randint(0, 10000000), rand_sp(),
                   rand_sp(), rand_sp())
