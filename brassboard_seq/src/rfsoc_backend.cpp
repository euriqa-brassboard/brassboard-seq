/*************************************************************************
 *   Copyright (c) 2024 - 2025 Yichao Yu <yyc1992@gmail.com>             *
 *                                                                       *
 *   This library is free software; you can redistribute it and/or       *
 *   modify it under the terms of the GNU Lesser General Public          *
 *   License as published by the Free Software Foundation; either        *
 *   version 3.0 of the License, or (at your option) any later version.  *
 *                                                                       *
 *   This library is distributed in the hope that it will be useful,     *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of      *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU    *
 *   Lesser General Public License for more details.                     *
 *                                                                       *
 *   You should have received a copy of the GNU Lesser General Public    *
 *   License along with this library. If not,                            *
 *   see <http://www.gnu.org/licenses/>.                                 *
 *************************************************************************/

#include "rfsoc_backend.h"

#include "rtval.h"

#include "event_time.h"
#include "utils.h"

#include <bit>
#include <bitset>
#include <sstream>
#include <utility>

#include <assert.h>

namespace brassboard_seq::rfsoc_backend {

struct output_flags_t {
    bool wait_trigger;
    bool sync;
    bool feedback_enable;
};

namespace {
#define WRAP_STR(name, str)                             \
    static inline PyObject *name##_str()                \
    {                                                   \
        static auto u = PyUnicode_FromString(str);      \
        return throw_if_not(u);                         \
    }
WRAP_STR(type, "type")
WRAP_STR(invalid, "invalid")
WRAP_STR(plut, "plut")
WRAP_STR(slut, "slut")
WRAP_STR(glut, "glut")
WRAP_STR(gseq, "gseq")
WRAP_STR(wait_anc, "wait_anc")
WRAP_STR(cont_anc, "cont_anc")
WRAP_STR(pulse_data, "pulse_data")
WRAP_STR(stream, "stream")

WRAP_STR(error, "error")
WRAP_STR(inst, "inst")
WRAP_STR(channel, "channel")
WRAP_STR(channels, "channels")
WRAP_STR(count, "count")
WRAP_STR(gaddrs, "gaddrs")
WRAP_STR(starts, "starts")
WRAP_STR(ends, "ends")
WRAP_STR(saddrs, "saddrs")
WRAP_STR(paddrs, "paddrs")
WRAP_STR(modtype, "modtype")
WRAP_STR(modtypes, "modtypes")

WRAP_STR(paddr, "paddr")
WRAP_STR(param, "param")
WRAP_STR(tone, "tone")
WRAP_STR(cycles, "cycles")
WRAP_STR(spline, "spline")
WRAP_STR(spline_mu, "spline_mu")
WRAP_STR(spline_shift, "spline_shift")
WRAP_STR(spline_freq, "spline_freq")
WRAP_STR(spline_amp, "spline_amp")
WRAP_STR(spline_phase, "spline_phase")

WRAP_STR(freq, "freq")
WRAP_STR(amp, "amp")
WRAP_STR(phase, "phase")
WRAP_STR(frame_rot, "frame_rot")
WRAP_STR(freq0, "freq0")
WRAP_STR(amp0, "amp0")
WRAP_STR(phase0, "phase0")
WRAP_STR(frame_rot0, "frame_rot0")
WRAP_STR(freq1, "freq1")
WRAP_STR(amp1, "amp1")
WRAP_STR(phase1, "phase1")
WRAP_STR(frame_rot1, "frame_rot1")

WRAP_STR(trig, "trig")
WRAP_STR(sync, "sync")
WRAP_STR(enable, "enable")
WRAP_STR(ff, "ff")
WRAP_STR(eof, "eof")
WRAP_STR(clr, "clr")
WRAP_STR(fwd, "fwd")
WRAP_STR(inv, "inv")

static void format_double(std::ostream &io, double v)
{
    // Unlike `operator<<`, which uses a fixed precision (6 by default),
    // `std::to_chars` of floating point number (no precision specified)
    // is guaranteed to use the shortest accurate representation
    // of the number.
    // With C++23, we could use `std::print(io, "{}", order)` instead.
    // (Not using std::format since GCC 11.1 for artiq-7 nix environment
    //  doesn't have it)
    char buff[64];
    auto ptr = to_chars(buff, v);
    io.write(buff, ptr - buff);
}

// Find the longest ranges that are at least min_val.
static void foreach_max_range_min_val(std::span<int> value, int min_val, auto &&cb)
{
    int N = value.size();
    int start_idx = -1;
    int start_val = 0;
    for (int i = 0; i < N; i++) {
        auto v = value[i];
        if (start_idx < 0) {
            // No start marked. Check if we should start a range.
            if (v >= min_val) {
                start_idx = i;
                start_val = v;
            }
            continue;
        }
        // We are above the start value, nothing to do
        if (v >= min_val) {
            start_val = std::min(start_val, v);
            continue;
        }
        cb(start_idx, i - 1, start_val);
        start_idx = -1;
    }
    if (start_idx >= 0) {
        cb(start_idx, N - 1, start_val);
    }
}

static inline int64_t seq_time_to_cycle(long long time)
{
    // Each cycle is 1 / 1ps / 409.6MHz sequence time unit
    // or 78125/32 sequence unit.
    constexpr auto numerator = 78125;
    constexpr auto denominator = 32;
    auto cycle_whole = time / numerator * denominator;
    auto cycle_frac = ((time % numerator) * denominator + numerator / 2) / numerator;
    return cycle_whole + cycle_frac;
}

static inline int64_t cycle_to_seq_time(int64_t cycle)
{
    constexpr auto numerator = 32;
    constexpr auto denominator = 78125;
    auto time_whole = cycle / numerator * denominator;
    auto time_frac = ((cycle % numerator) * denominator + numerator / 2) / numerator;
    return time_whole + time_frac;
}

static __attribute__((always_inline)) inline
cubic_spline_t spline_from_static(double v0)
{
    return { v0, 0, 0, 0 };
}

static __attribute__((optimize("-ffast-math"),always_inline)) inline
cubic_spline_t spline_from_values(double v0, double v1, double v2, double v3)
{
    // v = o0 + o1 * t + o2 * t^2 + o3 * t^3

    // v0 = o0
    // v1 = o0 + o1 / 3 + o2 / 9 + o3 / 27
    // v2 = o0 + o1 * 2 / 3 + o2 * 4 / 9 + o3 * 8 / 27
    // v3 = o0 + o1 + o2 + o3

    // o0 = v0
    // o1 = -5.5 * v0 + 9 * v1 - 4.5 * v2 + v3
    // o2 = 9 * v0 - 22.5 * v1 + 18 * v2 - 4.5 * v3
    // o3 = -4.5 * v0 + 13.5 * v1 - 13.5 * v2 + 4.5 * v3

    return {
        v0,
        -5.5 * v0 + 9 * v1 - 4.5 * v2 + v3,
        9 * v0 - 22.5 * v1 + 18 * v2 - 4.5 * v3,
        -4.5 * v0 + 13.5 * v1 - 13.5 * v2 + 4.5 * v3,
    };
}

static __attribute__((optimize("-ffast-math"),always_inline)) inline
double spline_eval(cubic_spline_t spline, double t)
{
    return spline.order0 + (spline.order1 +
                            (spline.order2 + spline.order3 * t) * t) * t;
}

static __attribute__((optimize("-ffast-math"),always_inline)) inline
cubic_spline_t spline_resample(cubic_spline_t spline, double t1, double t2)
{
    double dt = t2 - t1;
    double dt2 = dt * dt;
    double dt3 = dt2 * dt;
    double o3_3 = 3 * spline.order3;
    return {
        spline.order0 + (spline.order1 +
                         (spline.order2 + spline.order3 * t1) * t1) * t1,
        dt * (spline.order1 + (2 * spline.order2 + o3_3 * t1) * t1),
        dt2 * (spline.order2 + o3_3 * t1),
        dt3 * spline.order3,
    };
}

static inline cubic_spline_t approximate_spline(double v[5])
{
    double v0 = v[0];
    double v1 = v[1] * (2.0 / 3) + v[2] * (1.0 / 3);
    double v2 = v[3] * (2.0 / 3) + v[2] * (1.0 / 3);
    double v3 = v[4];
    // clamp v1 and v2 so that the numbers won't go too crazy
    if (v3 >= v0) {
        v1 = std::clamp(v1, v0, v3);
        v2 = std::clamp(v2, v0, v3);
    }
    else {
        v1 = std::clamp(v1, v3, v0);
        v2 = std::clamp(v2, v3, v0);
    }
    return spline_from_values(v0, v1, v2, v3);
}

static inline cubic_spline_t
spline_resample_cycle(cubic_spline_t sp, int64_t start, int64_t end,
                      int64_t cycle1, int64_t cycle2)
{
    if (cycle1 == start && cycle2 == end)
        return sp;
    return spline_resample(sp, double(cycle1 - start) / double(end - start),
                           double(cycle2 - start) / double(end - start));
}

struct IsFirst {
    bool first{true};
    bool get()
    {
        return std::exchange(first, false);
    }
};

}

struct TimedID {
    int64_t time;
    int16_t id;
};

static PyObject *rampfunctionbase_type;
static PyObject *seqcubicspline_type;

// order of the coefficient is order0, order1, order2, order3
template<int nbits> static constexpr inline std::pair<std::array<int64_t,4>,int>
convert_pdq_spline(std::array<double,4> sp, int64_t cycles, double scale)
{
    static_assert(nbits < 64);
    constexpr uint64_t mask = (uint64_t(1) << nbits) - 1;
    constexpr double bitscale = double(uint64_t(1) << nbits);

    std::array<int64_t,4> isp;
    // For the 0-th order, we can just round the number
    isp[0] = (__builtin_constant_p(sp[0]) && sp[0] == 0) ? 0 :
        round<int64_t>(sp[0] * (scale * bitscale)) & mask;
    if (sp[1] == 0 && sp[2] == 0 && sp[3] == 0)
        return {{isp[0], 0, 0, 0}, 0};

    double tstep = 1 / double(cycles);
    double tstep2 = tstep * tstep;
    double tstep3 = tstep2 * tstep;
    sp[1] = tstep * (sp[1] + sp[2] * tstep + sp[3] * tstep2);
    sp[2] = 2 * tstep2 * (sp[2] + 3 * sp[3] * tstep);
    sp[3] = 6 * tstep3 * sp[3];

    // See below, a threshold of `1 - 2^(-nbits)` should already be sufficient
    // to make sure the value doesn't round up.
    // Here we use `1 - 2 * 2^(-nbits)` just to be safe in case
    // we got the rounding mode wrong or sth...
    // FIXME: constexpr in C++23
    constexpr double round_thresh = 1 - 2 * std::ldexp(1, -nbits);

    // Now we'll map floating point values [-0.5, 0.5] to [-2^(nbits-1), 2^(nbits-1)]

    int shift_len = 31;

    int _exp[3];
    // For each higher order coefficients,
    // we need to figure out how much we can shift the coefficient left by
    // without overflow. For overflow values, we'll take the modulo
    // just like the 0-th order.
    for (int i = 1 ; i < 4; i++) {
        auto v = sp[i];
        if (v == 0) {
            _exp[i - 1] = 0;
            continue;
        }
        sp[i] = std::frexp(v * scale, &_exp[i - 1]);
        // Although frexp guarantees that `|fr| < 1`,
        // i.e. mathematically, `|fr| * 2^(nbits-1) < 2^(nbits-1)`,
        // or that it's smaller than the overflow threshold.
        // However, what we actually care about is to find the decomposition
        // so that `round(|fr| * 2^(nbits - 1)) < 2^(nbits - 1)`
        // to make sure it doesn't overflow during the conversion.

        // Here we catch the unlikely cases where `|fr| < 1`
        // will overflow during rounding.
        if (std::abs(sp[i]) > round_thresh) [[unlikely]] {
            // See above for not about the threshold.
            // This is about 2x the actual threshold
            // but the change of sub-optimal shift is still quite small.
            sp[i] /= 2;
            _exp[i - 1] += 1;
        }
        // Now we can shift this number up by at most `-_exp - 1` bits
        shift_len = std::min(shift_len, (-_exp[i - 1] - 1) / i);
    }

    if (shift_len < 0) [[unlikely]]
        shift_len = 0;

    isp[1] = round<int64_t>(std::ldexp(sp[1], _exp[0] + nbits + shift_len)) & mask;
    isp[2] = round<int64_t>(std::ldexp(sp[2], _exp[1] + nbits + shift_len * 2)) & mask;
    isp[3] = round<int64_t>(std::ldexp(sp[3], _exp[2] + nbits + shift_len * 3)) & mask;
    return { isp, shift_len };
}

constexpr static double output_clock = 819.2e6;

static constexpr inline std::pair<std::array<int64_t,4>,int>
convert_pdq_spline_freq(std::array<double,4> sp, int64_t cycles)
{
    return convert_pdq_spline<40>(sp, cycles, 1 / output_clock);
}

static constexpr inline std::pair<std::array<int64_t,4>,int>
convert_pdq_spline_amp(std::array<double,4> sp, int64_t cycles)
{
    auto [isp, shift_len] = convert_pdq_spline<17>(sp, cycles,
                                                   0.5 - std::ldexp(1, -17));
    for (auto &v: isp)
        v <<= 23;
    return { isp, shift_len };
}

static constexpr inline std::pair<std::array<int64_t,4>,int>
convert_pdq_spline_phase(std::array<double,4> sp, int64_t cycles)
{
    return convert_pdq_spline<40>(sp, cycles, 1);
}

using JaqalInst = Bits<int64_t,4>;

struct TimedInst {
    int64_t time;
    JaqalInst inst;
};

struct PulseAllocator {
    std::map<JaqalInst,int> pulses;

    void clear()
    {
        pulses.clear();
    }

    int get_addr(JaqalInst pulse)
    {
        auto [it, inserted] = pulses.emplace(pulse, int(pulses.size()));
        return it->second;
    }
};

struct PDQSpline {
    std::array<int64_t,4> orders;
    int shift;
    double scale;
    cubic_spline_t get_spline(uint64_t cycles) const
    {
        double order_scale = 1 / double(1ll << shift);
        double order_scale2 = order_scale * order_scale;
        std::array<double,4> forders;
        forders[0] = double(orders[0]) * scale;
        forders[1] = double(orders[1]) * (scale * order_scale);
        forders[2] = double(orders[2]) * (scale * order_scale2);
        forders[3] = double(orders[3]) * (scale * order_scale * order_scale2);

        double fcycles = double(cycles);
        double fcycles2 = fcycles * fcycles;

        forders[1] = (forders[1] - forders[2] / 2 + forders[3] / 3) * fcycles;
        forders[2] = (forders[2] - forders[3]) / 2 * fcycles2;
        forders[3] = forders[3] / 6 * fcycles2 * fcycles;

        return cubic_spline_t{ forders[0], forders[1], forders[2], forders[3] };
    }
};

static inline constexpr bool
test_bits(const JaqalInst &inst, unsigned b1, unsigned b2)
{
    return inst & JaqalInst::mask(b1, b2);
}

struct Jaqal_v1 {
    // All the instructions are 256 bits (32 bytes) and there are 7 types of
    // instructions roughly divided into 2 groups, programming and sequence output.
    // The two groups are determined by the GSEQ_ENABLE (247) bit,
    // where bit value 0 means programming mode and bit value 1 means
    // sequence output mode.
    // For either mode, the next two bits (PROG_MODE / SEQ_MODE) indicates
    // the exact type of the instruction and the meaning of the values is defined
    // in ProgMode and SeqMode respectively.
    enum class ProgMode: uint8_t {
        // 0 not used
        GLUT = 1, // Gate LUT
        SLUT = 2, // Sequence LUT
        PLUT = 3, // Pulse LUT
    };
    enum class SeqMode: uint8_t {
        // Sequence gate IDs
        GATE = 0,
        // Wait for Ancilla trigger, and sequence gate IDs based on readout
        WAIT_ANC = 1,
        // Continue sequencing based on previous ancilla data
        CONT_ANC = 2,
        // Bypass LUT/streaming mode
        STREAM = 3,
    };
    // Other than these bits, the only other bits thats shared by all types
    // of instructions appears to be the channel number bits [DMA_MUX+2, DMA_MUX].
    // The programming, sequencing as well as the lookup table storage appears to be
    // completely and statically segregated for each channels.

    // Both the STREAM instruction and the Pulse LUT instruction are used to
    // set the low level pulse/spline parameters and therefore shares very similar
    // format, i.e. the raw data format. The instruction consists of 6 parts,
    // |255       200|199       160|159 120|119  80|79   40|39    0|
    // |Metadata (56)|duration (40)|U3 (40)|U2 (40)|U1 (40)|U0 (40)|
    // where the duration and the U0-U3 fields specify the output parameters,
    // for a output parameter. The metadata part (bits [255, 200])
    // includes the shared bits mentioned above as well as other bits that
    // are meant for the spline engine/DDS.
    // The metadata format also depends on the type of parameters,
    // which is determined by the MODTYPE field (bits [255, 253]) and the values
    // are defined in ModType.
    enum class ModType: uint8_t {
        FRQMOD0 = 0,
        AMPMOD0 = 1,
        PHSMOD0 = 2,
        FRQMOD1 = 3,
        AMPMOD1 = 4,
        PHSMOD1 = 5,
        FRMROT0 = 6,
        FRMROT1 = 7,
    };
    // For the frequency/amplitude/phase parameters, the metadata format was
    // |255 253|252  248|247              245|244 241|240   229|
    // |MODTYPE|SPLSHIFT|GSEQ_ENABLE+SEQ_MODE|   0   |PLUT_ADDR|
    // |   228   |   227   |   226   |   225   |   224   |223|222 220|219 200|
    // |AMP_FB_EN|FRQ_FB_EN|OUTPUT_EN|SYNC_FLAG|WAIT_TRIG| 0 |DMA_MUX|   0   |

    // For the frame rotation parameter, the metadata format was
    // |255 253|252  248|247              245|244 241|240   229|
    // |MODTYPE|SPLSHIFT|GSEQ_ENABLE+SEQ_MODE|   0   |PLUT_ADDR|
    // |   228   |   227   |    226   |    225   |   224   |223|222 220|
    // |APPLY_EOF|CLR_FRAME|FWD_FRM_T1|FWD_FRM_T0|WAIT_TRIG| 0 |DMA_MUX|
    // |    219   |    218   |217 200|
    // |INV_FRM_T1|INV_FRM_T0|   0   |

    // The AMP_FB_EN, FRQ_FB_EN, OUTPUT_FB_EN, SYNC_FLAG, APPLY_EOF, CLR_FRAME,
    // FWD_FRM_T1, FWD_FRM_T0, INV_FRM_T1, INV_FRM_T0 are flags that affects
    // the behavior of the specific spline engines/DDS and are documented below.

    // SPLSHIFT is the shift that should be applied on the higher order terms
    // in the spline coefficient.
    // PLUT_ADDR is only used for the PLUT programming instruction and is the
    // 12 bit address to locate the pulse in the PLUT.
    // WAIT_TRIG is a flag to decide whether this command should wait for the trigger
    // to happen before being consumed by the spline engine.

    // The format for the SLUT programming instruction is,
    // |255 248|247               245|244 240|239  236|
    // |   0   |GSEQ_ENABLE+PROG_MODE|   0   |SLUT_CNT|
    // |228 223|222 220|219 216|215  204|203  192|
    // |   0   |DMA_MUX|   0   | SADDR8 | PADDR8 |
    // |191  180|179  168|167  156|155  144|143  132|131  120|119  108|107   96|
    // | SADDR7 | PADDR7 | SADDR6 | PADDR6 | SADDR5 | PADDR5 | SADDR4 | PADDR4 |
    // |95    84|83    72|71    60|59    48|47    36|35    24|23    12|11     0|
    // | SADDR3 | PADDR3 | SADDR2 | PADDR2 | SADDR1 | PADDR1 | SADDR0 | PADDR0 |

    // The format for the GLUT programming instruction is,
    // |255 248|247               245|244 239|238  236|228 223|222 220|219 216|
    // |   0   |GSEQ_ENABLE+PROG_MODE|   0   |GLUT_CNT|   0   |DMA_MUX|   0   |
    // |215  204|203  192|191 180|179  168|167  156|155 144|143  132|131  120|119 108|
    // | GADDR5 | START5 |  END5 | GADDR4 | START4 |  END4 | GADDR3 | START3 |  END3 |
    // |107   96|95    84|83  72|71    60|59    48|47  36|35    24|23    12|11   0|
    // | GADDR2 | START2 | END2 | GADDR1 | START1 | END1 | GADDR0 | START0 | END0 |

    // The format for the gate sequence instruction is,
    // |255 248|247              245|244  239|238 223|222 220|219 216|
    // |   0   |GSEQ_ENABLE+SEQ_MODE|GSEQ_CNT|   0   |DMA_MUX|   0   |
    // |215   207|206   198|197   189|188   180|179   171|170   162|161   153|152   144|
    // | GADDR23 | GADDR22 | GADDR21 | GADDR20 | GADDR19 | GADDR18 | GADDR17 | GADDR16 |
    // |143   135|134   126|125   117|116   108|107    99|98     90|89    81|80    72|
    // | GADDR15 | GADDR14 | GADDR13 | GADDR12 | GADDR11 | GADDR10 | GADDR9 | GADDR8 |
    // |71    63|62    54|53    45|44    36|35    27|26    18|17     9|8      0|
    // | GADDR7 | GADDR6 | GADDR5 | GADDR4 | GADDR3 | GADDR2 | GADDR1 | GADDR0 |

    // For all three formats, bit 0 to bit 219 (really, bit 215 due to the size of
    // the element) are packed with multiple elements of programming/sequence word
    // (12 + 12 bytes for SLUT programming, 12 + 12 * 2 bytes for GLUT programming
    // and 9 bytes for gate sequence). The number of elements packed
    // in the instruction is recorded in a count field (LSB at bit 236 for programming
    // and LSB at bit 239 for sequence instruction).

    struct Bits {
        // Modulation type (freq/phase/amp/framerot)
        static constexpr int MODTYPE = 253;

        // Fixed point shift for spline coefficients
        static constexpr int SPLSHIFT = 248;

        static constexpr int GSEQ_ENABLE = 247;
        static constexpr int PROG_MODE = 245;
        static constexpr int SEQ_MODE = 245;

        // Number of packed GLUT programming words
        static constexpr int GLUT_CNT = 236;
        // Number of packed SLUT programming words
        static constexpr int SLUT_CNT = 236;
        // Number of packed gate sequence identifiers
        static constexpr int GSEQ_CNT = 239;

        static constexpr int PLUT_ADDR = 229;

        //// For normal parameters
        // Amplitude feedback enable (placeholder)
        static constexpr int AMP_FB_EN = 228;
        // Frequency feedback enable
        static constexpr int FRQ_FB_EN = 227;
        // Toggle output enable
        static constexpr int OUTPUT_EN = 226;
        // Apply global synchronization
        static constexpr int SYNC_FLAG = 225;

        //// For frame rotation parameters
        // Apply frame rotation at end of pulse
        static constexpr int APPLY_EOF = 228;
        // Clear frame accumulator
        static constexpr int CLR_FRAME = 227;
        // Forward frame to tone 1/0
        static constexpr int FWD_FRM = 225;

        // Wait for external trigger
        static constexpr int WAIT_TRIG = 224;

        static constexpr int DMA_MUX = 220;
        static constexpr int PACKING_LIMIT = 220;

        //// Additional bits for frame rotation parameters
        // Invert sign on frame for tone 1/0
        static constexpr int INV_FRM = 218;

        // Start of metadata for raw data instruction
        static constexpr int METADATA = 200;
    };

    static constexpr inline auto
    pulse(uint64_t metadata, const std::array<int64_t,4> &isp, int64_t cycles)
    {
        assert((isp[0] >> 40) == 0);
        assert((isp[1] >> 40) == 0);
        assert((isp[2] >> 40) == 0);
        assert((isp[3] >> 40) == 0);
        assert((cycles >> 40) == 0);
        assume((isp[0] >> 40) == 0);
        assume((isp[1] >> 40) == 0);
        assume((isp[2] >> 40) == 0);
        assume((isp[3] >> 40) == 0);
        assume((cycles >> 40) == 0);

        std::array<int64_t,4> data{
            isp[0] | (isp[1] << 40),
            (isp[1] >> (64 - 40)) | (isp[2] << (80 - 64)) | (isp[3] << (120 - 64)),
            (isp[3] >> (128 - 120)) | (cycles << (160 - 128)),
            (cycles >> (192 - 160)) | int64_t(metadata << (200 - 192)),
        };
        return JaqalInst(data);
    }

    static constexpr inline uint64_t raw_param_metadata(
        ModType modtype, int channel, int shift_len, bool waittrig, bool sync,
        bool enable, bool fb_enable)
    {
        assert(shift_len >= 0 && shift_len < 32);
        assert(modtype != ModType::FRMROT0 && modtype != ModType::FRMROT1);
        uint64_t metadata = uint64_t(modtype) << (Bits::MODTYPE - Bits::METADATA);
        metadata |= uint64_t(shift_len) << (Bits::SPLSHIFT - Bits::METADATA);
        metadata |= uint64_t(channel) << (Bits::DMA_MUX - Bits::METADATA);
        metadata |= uint64_t(waittrig) << (Bits::WAIT_TRIG - Bits::METADATA);
        metadata |= uint64_t(enable) << (Bits::OUTPUT_EN - Bits::METADATA);
        metadata |= uint64_t(fb_enable) << (Bits::FRQ_FB_EN - Bits::METADATA);
        metadata |= uint64_t(sync) << (Bits::SYNC_FLAG - Bits::METADATA);
        return metadata;
    }
    static constexpr inline __attribute__((always_inline,flatten)) auto
    freq_pulse(int channel, int tone, cubic_spline_t sp, int64_t cycles, bool waittrig,
               bool sync, bool fb_enable)
    {
        assert(cycles >= 4);
        assert((cycles >> 40) == 0);
        assume(tone == 0 || tone == 1);
        auto [isp, shift_len] = convert_pdq_spline_freq(sp.to_array(), cycles);
        auto metadata = raw_param_metadata(tone ? ModType::FRQMOD1 : ModType::FRQMOD0,
                                           channel, shift_len, waittrig,
                                           sync, false, fb_enable);
        return pulse(metadata, isp, cycles);
    }
    static constexpr inline __attribute__((always_inline,flatten)) auto
    amp_pulse(int channel, int tone, cubic_spline_t sp, int64_t cycles, bool waittrig,
              bool sync=false, bool fb_enable=false)
    {
        assert(cycles >= 4);
        assert((cycles >> 40) == 0);
        assume(tone == 0 || tone == 1);
        auto [isp, shift_len] = convert_pdq_spline_amp(sp.to_array(), cycles);
        auto metadata = raw_param_metadata(tone ? ModType::AMPMOD1 : ModType::AMPMOD0,
                                           channel, shift_len, waittrig,
                                           sync, false, fb_enable);
        return pulse(metadata, isp, cycles);
    }
    static constexpr inline __attribute__((always_inline,flatten)) auto
    phase_pulse(int channel, int tone, cubic_spline_t sp, int64_t cycles, bool waittrig,
                bool sync=false, bool fb_enable=false)
    {
        assert(cycles >= 4);
        assert((cycles >> 40) == 0);
        assume(tone == 0 || tone == 1);
        auto [isp, shift_len] = convert_pdq_spline_phase(sp.to_array(), cycles);
        auto metadata = raw_param_metadata(tone ? ModType::PHSMOD1 : ModType::PHSMOD0,
                                           channel, shift_len, waittrig,
                                           sync, false, fb_enable);
        return pulse(metadata, isp, cycles);
    }

    static constexpr inline uint64_t raw_frame_metadata(
        ModType modtype, int channel, int shift_len, bool waittrig,
        bool apply_at_end, bool rst_frame, int fwd_frame_mask, int inv_frame_mask)
    {
        assert(shift_len >= 0 && shift_len < 32);
        assert(modtype == ModType::FRMROT0 || modtype == ModType::FRMROT1);
        uint64_t metadata = uint64_t(modtype) << (Bits::MODTYPE - Bits::METADATA);
        metadata |= uint64_t(shift_len) << (Bits::SPLSHIFT - Bits::METADATA);
        metadata |= uint64_t(channel) << (Bits::DMA_MUX - Bits::METADATA);
        metadata |= uint64_t(waittrig) << (Bits::WAIT_TRIG - Bits::METADATA);
        metadata |= uint64_t(apply_at_end) << (Bits::APPLY_EOF - Bits::METADATA);
        metadata |= uint64_t(rst_frame) << (Bits::CLR_FRAME - Bits::METADATA);
        metadata |= uint64_t(fwd_frame_mask) << (Bits::FWD_FRM - Bits::METADATA);
        metadata |= uint64_t(inv_frame_mask) << (Bits::INV_FRM - Bits::METADATA);
        return metadata;
    }
    static constexpr inline __attribute__((always_inline,flatten)) auto
    frame_pulse(int channel, int tone, cubic_spline_t sp, int64_t cycles,
                bool waittrig, bool apply_at_end, bool rst_frame,
                int fwd_frame_mask, int inv_frame_mask)
    {
        assert(cycles >= 4);
        assert((cycles >> 40) == 0);
        assume(tone == 0 || tone == 1);
        auto [isp, shift_len] = convert_pdq_spline_phase(sp.to_array(), cycles);
        auto metadata = raw_frame_metadata(tone ? ModType::FRMROT1 : ModType::FRMROT0,
                                           channel, shift_len, waittrig, apply_at_end,
                                           rst_frame, fwd_frame_mask, inv_frame_mask);
        return pulse(metadata, isp, cycles);
    }

    // LUT Address Widths
    // These values contain the number of bits used for an address for a
    // particular LUT. The address width is the same for reading and writing
    // data for the Pulse LUT (PLUT) and the Sequence LUT (SLUT), but the
    // address width is asymmetric for the Gate LUT (GLUT). This is because
    // the read address is partly completed by external hardware inputs and
    // the read address size (GLUTW) is thus smaller than the write address
    // size (GPRGW) used to program the GLUT
    static constexpr int GPRGW = 12;  // Gate LUT write address width
    static constexpr int GLUTW = 9;  // Gate LUT read address width
    static constexpr int SLUTW = 12;  // Sequence LUT address width
    static constexpr int PLUTW = 12;  // Pulse LUT address width

    static constexpr int SLUT_ELSZ = SLUTW + PLUTW;
    static constexpr int GLUT_ELSZ = GPRGW + 2 * SLUTW;
    static constexpr int GSEQ_ELSZ = GLUTW;
    // Number of programming or gate sequence words that can be packed into a single
    // transfer. PLUT programming data is always one word per transfer.
    static constexpr int SLUT_MAXCNT = Bits::PACKING_LIMIT / SLUT_ELSZ;
    static constexpr int GLUT_MAXCNT = Bits::PACKING_LIMIT / GLUT_ELSZ;
    static constexpr int GSEQ_MAXCNT = Bits::PACKING_LIMIT / GSEQ_ELSZ;

    static constexpr inline JaqalInst stream(JaqalInst pulse)
    {
        pulse |= JaqalInst(uint8_t(SeqMode::STREAM)) << Bits::SEQ_MODE;
        pulse |= JaqalInst(1) << Bits::GSEQ_ENABLE;
        return pulse;
    }

    static constexpr inline JaqalInst program_PLUT(JaqalInst pulse, uint16_t addr)
    {
        assert((addr >> PLUTW) == 0);
        pulse |= JaqalInst(uint8_t(ProgMode::PLUT)) << Bits::PROG_MODE;
        pulse |= JaqalInst(addr) << Bits::PLUT_ADDR;
        return pulse;
    }

    static constexpr inline auto
    program_SLUT(uint8_t chn, const uint16_t *saddrs, const uint16_t *paddrs, int n)
    {
        JaqalInst inst;
        assert(n <= SLUT_MAXCNT);
        for (int i = 0; i < n; i++) {
            assert((paddrs[i] >> PLUTW) == 0);
            inst |= JaqalInst(paddrs[i]) << SLUT_ELSZ * i;
            assert((saddrs[i] >> SLUTW) == 0);
            inst |= JaqalInst(saddrs[i]) << (SLUT_ELSZ * i + PLUTW);
        }
        inst |= JaqalInst(uint8_t(ProgMode::SLUT)) << Bits::PROG_MODE;
        inst |= JaqalInst(uint8_t(n)) << Bits::SLUT_CNT;
        inst |= JaqalInst(chn) << Bits::DMA_MUX;
        return inst;
    }

    static constexpr inline auto
    program_GLUT(uint8_t chn, const uint16_t *gaddrs, const uint16_t *starts,
                 const uint16_t *ends, int n)
    {
        JaqalInst inst;
        assert(n <= GLUT_MAXCNT);
        for (int i = 0; i < n; i++) {
            assert((gaddrs[i] >> GPRGW) == 0);
            inst |= JaqalInst(gaddrs[i]) << (GLUT_ELSZ * i + SLUTW * 2);
            assert((ends[i] >> SLUTW) == 0);
            inst |= JaqalInst(ends[i]) << (GLUT_ELSZ * i + SLUTW);
            assert((starts[i] >> SLUTW) == 0);
            inst |= JaqalInst(starts[i]) << (GLUT_ELSZ * i);
        }
        inst |= JaqalInst(uint8_t(ProgMode::GLUT)) << Bits::PROG_MODE;
        inst |= JaqalInst(uint8_t(n)) << Bits::GLUT_CNT;
        inst |= JaqalInst(chn) << Bits::DMA_MUX;
        return inst;
    }

    static constexpr inline auto
    sequence(uint8_t chn, SeqMode m, uint16_t *gaddrs, int n)
    {
        JaqalInst inst;
        assert(n <= GSEQ_MAXCNT);
        assert(m != SeqMode::STREAM);
        for (int i = 0; i < n; i++) {
            assert((gaddrs[i] >> GLUTW) == 0);
            inst |= JaqalInst(gaddrs[i]) << (GSEQ_ELSZ * i);
        }
        inst |= JaqalInst(uint8_t(m)) << Bits::SEQ_MODE;
        inst |= JaqalInst(1) << Bits::GSEQ_ENABLE;
        inst |= JaqalInst(n) << Bits::GSEQ_CNT;
        inst |= JaqalInst(chn) << Bits::DMA_MUX;
        return inst;
    }

    static constexpr inline int get_chn(const JaqalInst &inst)
    {
        return (inst >> Bits::DMA_MUX)[0] & 0x7;
    }

    struct Executor {
        enum class Error {
            Reserved,
            GLUT_OOB,
            SLUT_OOB,
            GSEQ_OOB,
        };
        static const char *error_msg(Error err)
        {
            switch (err) {
            default:
            case Error::Reserved:
                return "reserved";
            case Error::GLUT_OOB:
                return "glut_oob";
            case Error::SLUT_OOB:
                return "slut_oob";
            case Error::GSEQ_OOB:
                return "gseq_oob";
            }
        }
        struct PulseTarget {
            enum Type {
                None,
                PLUT,
                Stream,
            } type;
            uint16_t addr;
        };
        enum class ParamType {
            Freq,
            Amp,
            Phase,
        };
        template<typename T>
        static void execute(auto &&cb, const std::span<T> insts, bool allow_op0=false)
        {
            auto sz = insts.size_bytes();
            if (sz % sizeof(JaqalInst) != 0)
                throw std::invalid_argument("Instruction stream length "
                                            "not a multiple of instruction size");
            auto p = (const char*)insts.data();
            for (size_t i = 0; i < sz; i += sizeof(JaqalInst)) {
                JaqalInst inst;
                memcpy(&inst, p + i, sizeof(JaqalInst));
                if (i != 0)
                    cb.next();
                execute(cb, inst, allow_op0);
            }
        }
        static void execute(auto &&cb, const JaqalInst &inst, bool allow_op0=true)
        {
            int op = (inst >> Bits::PROG_MODE)[0] & 0x7;
            switch (op) {
            default:
            case 0:
                if (!allow_op0) {
                    invalid(cb, inst, Error::Reserved);
                    return;
                }
                pulse(cb, inst, PulseTarget::None);
                return;
            case int(ProgMode::PLUT):
                pulse(cb, inst, PulseTarget::PLUT);
                return;
            case int(SeqMode::STREAM) | 4:
                pulse(cb, inst, PulseTarget::Stream);
                return;

            case int(ProgMode::GLUT):
                GLUT(cb, inst);
                return;
            case int(ProgMode::SLUT):
                SLUT(cb, inst);
                return;
            case int(SeqMode::GATE) | 4:
            case int(SeqMode::WAIT_ANC) | 4:
            case int(SeqMode::CONT_ANC) | 4:
                GSEQ(cb, inst, SeqMode(op & 3));
                return;
            }
        }
    private:
        static void invalid(auto &&cb, const JaqalInst &inst, Error err)
        {
            cb.invalid(inst, err);
        }
        static void GLUT(auto &&cb, const JaqalInst &inst)
        {
            if (test_bits(inst, 248, 255) || test_bits(inst, 239, 244) ||
                test_bits(inst, 223, 228)) {
                invalid(cb, inst, Error::Reserved);
                return;
            }
            int cnt = (inst >> Bits::GLUT_CNT)[0] & 0x7;
            int chn = get_chn(inst);
            if (cnt > GLUT_MAXCNT) {
                invalid(cb, inst, Error::GLUT_OOB);
                return;
            }
            else if (test_bits(inst, GLUT_ELSZ * cnt, 219)) {
                invalid(cb, inst, Error::Reserved);
                return;
            }
            uint16_t gaddrs[GLUT_MAXCNT];
            uint16_t starts[GLUT_MAXCNT];
            uint16_t ends[GLUT_MAXCNT];
            for (int i = 0; i < cnt; i++) {
                auto w = (inst >> GLUT_ELSZ * i)[0];
                gaddrs[i] = (w >> (SLUTW * 2)) & ((1 << GPRGW) - 1);
                ends[i] = (w >> SLUTW) & ((1 << SLUTW) - 1);
                starts[i] = w & ((1 << SLUTW) - 1);
            }
            cb.GLUT(chn, gaddrs, starts, ends, cnt);
        }

        static void SLUT(auto &&cb, const JaqalInst &inst)
        {
            if (test_bits(inst, 248, 255) || test_bits(inst, 240, 244) ||
                test_bits(inst, 223, 228)) {
                invalid(cb, inst, Error::Reserved);
                return;
            }
            int cnt = (inst >> Bits::SLUT_CNT)[0] & 0xf;
            int chn = get_chn(inst);
            if (cnt > SLUT_MAXCNT) {
                invalid(cb, inst, Error::SLUT_OOB);
                return;
            }
            else if (test_bits(inst, SLUT_ELSZ * cnt, 219)) {
                invalid(cb, inst, Error::Reserved);
                return;
            }
            uint16_t saddrs[SLUT_MAXCNT];
            uint16_t paddrs[SLUT_MAXCNT];
            for (int i = 0; i < cnt; i++) {
                auto w = (inst >> SLUT_ELSZ * i)[0];
                saddrs[i] = (w >> PLUTW) & ((1 << SLUTW) - 1);
                paddrs[i] = w & ((1 << PLUTW) - 1);
            }
            cb.SLUT(chn, saddrs, paddrs, cnt);
        }

        static void GSEQ(auto &&cb, const JaqalInst &inst, SeqMode m)
        {
            if (test_bits(inst, 248, 255) || test_bits(inst, 223, 238)) {
                invalid(cb, inst, Error::Reserved);
                return;
            }
            int cnt = (inst >> Bits::GSEQ_CNT)[0] & 0x3f;
            int chn = get_chn(inst);
            if (cnt > GSEQ_MAXCNT) {
                invalid(cb, inst, Error::GSEQ_OOB);
                return;
            }
            else if (test_bits(inst, GSEQ_ELSZ * cnt, 219)) {
                invalid(cb, inst, Error::Reserved);
                return;
            }
            uint16_t gaddrs[GSEQ_MAXCNT];
            for (int i = 0; i < cnt; i++) {
                auto w = (inst >> GSEQ_ELSZ * i)[0];
                gaddrs[i] = w & ((1 << GLUTW) - 1);
            }
            cb.GSEQ(chn, gaddrs, cnt, m);
        }

        static void pulse(auto &&cb, const JaqalInst &inst, PulseTarget::Type type)
        {
            uint16_t addr = (inst >> Bits::PLUT_ADDR)[0] & ((1 << PLUTW) - 1);
            if (type != PulseTarget::PLUT && addr) {
                invalid(cb, inst, Error::Reserved);
                return;
            }
            PulseTarget tgt{type, addr};
            if (test_bits(inst, 241, 244) || test_bits(inst, 223, 223)) {
                invalid(cb, inst, Error::Reserved);
                return;
            }
            int chn = get_chn(inst);
            PDQSpline spl;
            spl.shift = (inst >> Bits::SPLSHIFT)[0] & 0x1f;
            auto load_s40 = [&] (int offset) {
                auto mask = (int64_t(1) << 40) - 1;
                auto data = (inst >> offset)[0] & mask;
                if (data & (int64_t(1) << 39))
                    return data | ~mask;
                return data;
            };
            spl.orders[0] = load_s40(40 * 0);
            spl.orders[1] = load_s40(40 * 1);
            spl.orders[2] = load_s40(40 * 2);
            spl.orders[3] = load_s40(40 * 3);
            int64_t cycles = (inst >> 40 * 4)[0] & ((int64_t(1) << 40) - 1);
            switch (ModType((inst >> Bits::MODTYPE)[0] & 0x7)) {
            case ModType::FRQMOD0:
                spl.scale = output_clock / double(1ll << 40);
                param_pulse(cb, inst, chn, 0, ParamType::Freq, spl, cycles, tgt);
                return;
            case ModType::AMPMOD0:
                spl.scale = 1 / double(((1ll << 16) - 1ll) << 23);
                param_pulse(cb, inst, chn, 0, ParamType::Amp, spl, cycles, tgt);
                return;
            case ModType::PHSMOD0:
                spl.scale = 1 / double(1ll << 40);
                param_pulse(cb, inst, chn, 0, ParamType::Phase, spl, cycles, tgt);
                return;
            case ModType::FRQMOD1:
                spl.scale = output_clock / double(1ll << 40);
                param_pulse(cb, inst, chn, 1, ParamType::Freq, spl, cycles, tgt);
                return;
            case ModType::AMPMOD1:
                spl.scale = 1 / double(((1ll << 16) - 1ll) << 23);
                param_pulse(cb, inst, chn, 1, ParamType::Amp, spl, cycles, tgt);
                return;
            case ModType::PHSMOD1:
                spl.scale = 1 / double(1ll << 40);
                param_pulse(cb, inst, chn, 1, ParamType::Phase, spl, cycles, tgt);
                return;
            case ModType::FRMROT0:
                spl.scale = 1 / double(1ll << 40);
                frame_pulse(cb, inst, chn, 0, spl, cycles, tgt);
                return;
            case ModType::FRMROT1:
                spl.scale = 1 / double(1ll << 40);
                frame_pulse(cb, inst, chn, 1, spl, cycles, tgt);
                return;
            }
        }
        static void param_pulse(auto &&cb, const JaqalInst &inst, int chn, int tone,
                                ParamType param, const PDQSpline &spl, int64_t cycles,
                                PulseTarget tgt)
        {
            if (test_bits(inst, 200, 219) ||
                test_bits(inst, Bits::AMP_FB_EN, Bits::AMP_FB_EN)) {
                invalid(cb, inst, Error::Reserved);
                return;
            }
            bool fb = (inst >> Bits::FRQ_FB_EN)[0] & 1;
            bool en = (inst >> Bits::OUTPUT_EN)[0] & 1;
            bool sync = (inst >> Bits::SYNC_FLAG)[0] & 1;
            bool trig = (inst >> Bits::WAIT_TRIG)[0] & 1;
            cb.param_pulse(chn, tone, param, spl, cycles, trig, sync, en, fb, tgt);

        }
        static void frame_pulse(auto &&cb, const JaqalInst &inst, int chn, int tone,
                                const PDQSpline &spl, int64_t cycles, PulseTarget tgt)
        {
            if (test_bits(inst, 200, 217)) {
                invalid(cb, inst, Error::Reserved);
                return;
            }
            bool trig = (inst >> Bits::WAIT_TRIG)[0] & 1;
            bool apply_eof = (inst >> Bits::APPLY_EOF)[0] & 1;
            bool clr_frame = (inst >> Bits::CLR_FRAME)[0] & 1;
            int fwd_frame_mask = (inst >> Bits::FWD_FRM)[0] & 3;
            int inv_frame_mask = (inst >> Bits::INV_FRM)[0] & 3;
            cb.frame_pulse(chn, tone, spl, cycles, trig, apply_eof, clr_frame,
                           fwd_frame_mask, inv_frame_mask, tgt);
        }
    };

    struct Printer {
        void invalid(const JaqalInst &inst, Executor::Error err)
        {
            io << "invalid(" << Executor::error_msg(err) << "): "
               << std::noshowbase << inst;
        }

        void next()
        {
            io << std::endl;
        }

        void GLUT(uint8_t chn, const uint16_t *gaddrs, const uint16_t *starts,
                  const uint16_t *ends, int cnt)
        {
            io << "glut." << int(chn) << "[";
            io << cnt << "]";
            for (int i = 0; i < cnt; i++) {
                io << " [" << gaddrs[i] << "]=[" << starts[i] << "," << ends[i] << "]";
            }
        }
        void SLUT(uint8_t chn, const uint16_t *saddrs, const uint16_t *paddrs, int cnt)
        {
            io << "slut." << int(chn) << "[";
            io << cnt << "]";
            for (int i = 0; i < cnt; i++) {
                io << " [" << saddrs[i] << "]=" << paddrs[i];
            }
        }
        void GSEQ(uint8_t chn, const uint16_t *gaddrs, int cnt, SeqMode m)
        {
            if (m == SeqMode::GATE) {
                io << "gseq.";
            }
            else if (m == SeqMode::WAIT_ANC) {
                io << "wait_anc.";
            }
            else if (m == SeqMode::CONT_ANC) {
                io << "cont_anc.";
            }
            io << int(chn) << "[";
            io << cnt << "]";
            for (int i = 0; i < cnt; i++) {
                io << " " << gaddrs[i];
            }
        }

        void param_pulse(int chn, int tone, Executor::ParamType param,
                         const PDQSpline &spl, int64_t cycles, bool waittrig,
                         bool sync, bool enable, bool fb_enable,
                         Executor::PulseTarget tgt)
        {
            print_pulse_prefix(tgt, (param == Executor::ParamType::Freq ? "freq" :
                                     (param == Executor::ParamType::Amp ? "amp" :
                                      "phase")), chn, tone, cycles, spl);
            if (waittrig)
                io << " trig";
            if (sync)
                io << " sync";
            if (enable)
                io << " enable";
            if (fb_enable)
                io << " ff";
        }
        void frame_pulse(int chn, int tone, const PDQSpline &spl, int64_t cycles,
                         bool waittrig, bool apply_eof, bool clr_frame,
                         int fwd_frame_mask, int inv_frame_mask,
                         Executor::PulseTarget tgt)
        {
            print_pulse_prefix(tgt, "frame_rot", chn, tone, cycles, spl);
            if (waittrig)
                io << " trig";
            if (apply_eof)
                io << " eof";
            if (clr_frame)
                io << " clr";
            io << " fwd:" << std::dec << std::noshowbase << fwd_frame_mask
               << " inv:" << std::dec << std::noshowbase << inv_frame_mask;
        }

        std::ostream &io;
        bool print_float{true};

    private:
        void print_pulse_prefix(Executor::PulseTarget tgt, const char *name,
                                int chn, int tone, int64_t cycles,
                                const PDQSpline &spl)
        {
            if (tgt.type == Executor::PulseTarget::None) {
                io << "pulse_data.";
            }
            else if (tgt.type == Executor::PulseTarget::PLUT) {
                io << "plut.";
            }
            else if (tgt.type == Executor::PulseTarget::Stream) {
                io << "stream.";
            }
            io << chn << " ";
            if (tgt.type == Executor::PulseTarget::PLUT)
                io << "[" << tgt.addr << "]=";
            io << name << tone << " <" << cycles << "> {";
            auto print_orders = [&] (auto order1, auto order2, auto order3, auto cb) {
                if (order1 || order2 || order3)
                    cb(1, order1);
                if (order2 || order3)
                    cb(2, order2);
                if (order3)
                    cb(3, order3);
            };
            if (print_float) {
                auto cspl = spl.get_spline(cycles);
                format_double(io, cspl.order0);
                print_orders(cspl.order1, cspl.order2, cspl.order3,
                             [&] (int, auto order) { format_double(io << ", ", order); });
            }
            else {
                io << std::showbase << std::hex << spl.orders[0];
                print_orders(spl.orders[1], spl.orders[2], spl.orders[3],
                             [&] (int i, auto order) {
                                 io << ", " << std::showbase << std::hex << order;
                                 if (spl.shift) {
                                     io << ">>" << std::dec << (spl.shift * i);
                                 }
                             });
            }
            io << "}";
        }
    };

    struct DictConverter {
        void invalid(const JaqalInst &inst, Executor::Error err)
        {
            throw_if(PyDict_SetItem(dict, type_str(), invalid_str()));
            py_object msg(pyunicode_from_string(Executor::error_msg(err)));
            throw_if(PyDict_SetItem(dict, error_str(), msg));
            std::ostringstream stm;
            stm << inst;
            auto str = stm.str();
            py_object py_str(pyunicode_from_string(str.c_str()));
            throw_if(PyDict_SetItem(dict, inst_str(), py_str));
        }

        void GLUT(uint8_t chn, const uint16_t *gaddrs, const uint16_t *starts,
                  const uint16_t *ends, int cnt)
        {
            throw_if(PyDict_SetItem(dict, type_str(), glut_str()));
            throw_if(PyDict_SetItem(dict, channel_str(), pylong_cached(chn)));
            throw_if(PyDict_SetItem(dict, count_str(), pylong_cached(cnt)));
            py_object py_gaddrs(pylist_new(cnt));
            py_object py_starts(pylist_new(cnt));
            py_object py_ends(pylist_new(cnt));
            for (int i = 0; i < cnt; i++) {
                PyList_SET_ITEM(py_gaddrs.get(), i, pylong_from_long(gaddrs[i]));
                PyList_SET_ITEM(py_starts.get(), i, pylong_from_long(starts[i]));
                PyList_SET_ITEM(py_ends.get(), i, pylong_from_long(ends[i]));
            }
            throw_if(PyDict_SetItem(dict, gaddrs_str(), py_gaddrs));
            throw_if(PyDict_SetItem(dict, starts_str(), py_starts));
            throw_if(PyDict_SetItem(dict, ends_str(), py_ends));
        }

        void SLUT(uint8_t chn, const uint16_t *saddrs, const uint16_t *paddrs, int cnt)
        {
            throw_if(PyDict_SetItem(dict, type_str(), slut_str()));
            throw_if(PyDict_SetItem(dict, channel_str(), pylong_cached(chn)));
            throw_if(PyDict_SetItem(dict, count_str(), pylong_cached(cnt)));
            py_object py_saddrs(pylist_new(cnt));
            py_object py_paddrs(pylist_new(cnt));
            for (int i = 0; i < cnt; i++) {
                PyList_SET_ITEM(py_saddrs.get(), i, pylong_from_long(saddrs[i]));
                PyList_SET_ITEM(py_paddrs.get(), i, pylong_from_long(paddrs[i]));
            }
            throw_if(PyDict_SetItem(dict, saddrs_str(), py_saddrs));
            throw_if(PyDict_SetItem(dict, paddrs_str(), py_paddrs));
        }
        void GSEQ(uint8_t chn, const uint16_t *gaddrs, int cnt, SeqMode m)
        {
            if (m == SeqMode::GATE) {
                throw_if(PyDict_SetItem(dict, type_str(), gseq_str()));
            }
            else if (m == SeqMode::WAIT_ANC) {
                throw_if(PyDict_SetItem(dict, type_str(), wait_anc_str()));
            }
            else if (m == SeqMode::CONT_ANC) {
                throw_if(PyDict_SetItem(dict, type_str(), cont_anc_str()));
            }
            throw_if(PyDict_SetItem(dict, channel_str(), pylong_cached(chn)));
            throw_if(PyDict_SetItem(dict, count_str(), pylong_cached(cnt)));
            py_object py_gaddrs(pylist_new(cnt));
            for (int i = 0; i < cnt; i++)
                PyList_SET_ITEM(py_gaddrs.get(), i, pylong_from_long(gaddrs[i]));
            throw_if(PyDict_SetItem(dict, gaddrs_str(), py_gaddrs));
        }

        void param_pulse(int chn, int tone, Executor::ParamType param,
                         const PDQSpline &spl, int64_t cycles, bool waittrig,
                         bool sync, bool enable, bool fb_enable,
                         Executor::PulseTarget tgt)
        {
            pulse_to_dict(tgt, (param == Executor::ParamType::Freq ? freq_str() :
                                (param == Executor::ParamType::Amp ? amp_str() :
                                 phase_str())), chn, tone, cycles, spl);
            throw_if(PyDict_SetItem(dict, trig_str(), waittrig ? Py_True : Py_False));
            throw_if(PyDict_SetItem(dict, sync_str(), sync ? Py_True : Py_False));
            throw_if(PyDict_SetItem(dict, enable_str(), enable ? Py_True : Py_False));
            throw_if(PyDict_SetItem(dict, ff_str(), fb_enable ? Py_True : Py_False));
        }
        void frame_pulse(int chn, int tone, const PDQSpline &spl, int64_t cycles,
                         bool waittrig, bool apply_eof, bool clr_frame,
                         int fwd_frame_mask, int inv_frame_mask,
                         Executor::PulseTarget tgt)
        {
            pulse_to_dict(tgt, frame_rot_str(), chn, tone, cycles, spl);
            throw_if(PyDict_SetItem(dict, trig_str(), waittrig ? Py_True : Py_False));
            throw_if(PyDict_SetItem(dict, eof_str(), apply_eof ? Py_True : Py_False));
            throw_if(PyDict_SetItem(dict, clr_str(), clr_frame ? Py_True : Py_False));
            throw_if(PyDict_SetItem(dict, fwd_str(), pylong_cached(fwd_frame_mask)));
            throw_if(PyDict_SetItem(dict, inv_str(), pylong_cached(inv_frame_mask)));
        }

        py_object dict{pydict_new()};

    private:
        void pulse_to_dict(Executor::PulseTarget tgt, PyObject *name, int chn,
                           int tone, int64_t cycles, const PDQSpline &spl)
        {
            if (tgt.type == Executor::PulseTarget::None) {
                throw_if(PyDict_SetItem(dict, type_str(), pulse_data_str()));
            }
            else if (tgt.type == Executor::PulseTarget::PLUT) {
                throw_if(PyDict_SetItem(dict, type_str(), plut_str()));
            }
            else if (tgt.type == Executor::PulseTarget::Stream) {
                throw_if(PyDict_SetItem(dict, type_str(), stream_str()));
            }
            throw_if(PyDict_SetItem(dict, channel_str(), pylong_cached(chn)));
            if (tgt.type == Executor::PulseTarget::PLUT) {
                py_object paddr(pylong_from_long(tgt.addr));
                throw_if(PyDict_SetItem(dict, paddr_str(), paddr));
            }
            throw_if(PyDict_SetItem(dict, param_str(), name));
            throw_if(PyDict_SetItem(dict, tone_str(), pylong_cached(tone)));
            py_object py_cycles(pylong_from_longlong(cycles));
            throw_if(PyDict_SetItem(dict, cycles_str(), py_cycles));
            py_object py_spl(pylist_new(4));
            for (int i = 0; i < 4; i++)
                PyList_SET_ITEM(py_spl.get(), i, pylong_from_longlong(spl.orders[i]));
            throw_if(PyDict_SetItem(dict, spline_mu_str(), py_spl));
            throw_if(PyDict_SetItem(dict, spline_shift_str(), pylong_cached(spl.shift)));
            auto fspl = spl.get_spline(cycles);
            py_object py_fspl(pylist_new(4));
            PyList_SET_ITEM(py_fspl.get(), 0, pyfloat_from_double(fspl.order0));
            PyList_SET_ITEM(py_fspl.get(), 1, pyfloat_from_double(fspl.order1));
            PyList_SET_ITEM(py_fspl.get(), 2, pyfloat_from_double(fspl.order2));
            PyList_SET_ITEM(py_fspl.get(), 3, pyfloat_from_double(fspl.order3));
            throw_if(PyDict_SetItem(dict, spline_str(), py_fspl));
        }
    };

    struct PulseSequencer {
        PulseSequencer()
        {
            reset();
        }

        void reset()
        {
            memset(_channel_mem, 0xff, sizeof(channels));
        }

        void invalid(const JaqalInst&, Executor::Error)
        {
            pulses.push_back(JaqalInst::mask(0, 255));
        }

        void next()
        {
        }

        void GLUT(uint8_t chn, const uint16_t *gaddrs, const uint16_t *starts,
                  const uint16_t *ends, int cnt)
        {
            auto &glut = channels[chn].glut;
            for (int i = 0; i < cnt; i++) {
                glut[gaddrs[i]] = { starts[i], ends[i] };
            }
        }
        void SLUT(uint8_t chn, const uint16_t *saddrs, const uint16_t *paddrs, int cnt)
        {
            auto &slut = channels[chn].slut;
            for (int i = 0; i < cnt; i++) {
                slut[saddrs[i]] = paddrs[i];
            }
        }
        void GSEQ(uint8_t chn, const uint16_t *gaddrs, int cnt, SeqMode m)
        {
            if (m != SeqMode::GATE) {
                pulses.push_back(JaqalInst::mask(0, 255));
                return;
            }
            auto &channel = channels[chn];
            for (auto gaddr: std::span(gaddrs, cnt)) {
                auto [start, end] = channel.glut[gaddr];
                if (start >= 4096 || end >= 4096 || end < start) {
                    pulses.push_back(JaqalInst::mask(0, 255));
                    continue;
                }
                for (auto i = start; i <= end; i++) {
                    auto paddr = channel.slut[i];
                    if (paddr >= 4096) {
                        pulses.push_back(JaqalInst::mask(0, 255));
                    }
                    else {
                        pulses.push_back(channel.plut[paddr]);
                    }
                }
            }
        }

        void param_pulse(int chn, int tone, Executor::ParamType param,
                         const PDQSpline &spl, int64_t cycles, bool waittrig,
                         bool sync, bool enable, bool fb_enable,
                         Executor::PulseTarget tgt)
        {
            auto metadata =
                raw_param_metadata(ModType(tone * 3 + int(param)), chn, spl.shift,
                                   waittrig, sync, enable, fb_enable);
            process_pulse(chn, metadata, spl.orders, cycles, tgt);
        }
        void frame_pulse(int chn, int tone, const PDQSpline &spl, int64_t cycles,
                         bool waittrig, bool apply_eof, bool clr_frame,
                         int fwd_frame_mask, int inv_frame_mask,
                         Executor::PulseTarget tgt)
        {
            auto metadata =
                raw_frame_metadata(tone ? ModType::FRMROT1 : ModType::FRMROT0,
                                   chn, spl.shift, waittrig, apply_eof, clr_frame,
                                   fwd_frame_mask, inv_frame_mask);
            process_pulse(chn, metadata, spl.orders, cycles, tgt);
        }

        std::vector<JaqalInst> pulses;
    private:
        void process_pulse(int chn, uint64_t metadata, const std::array<int64_t,4> &_sp,
                           int64_t cycles, Executor::PulseTarget tgt)
        {
            std::array<int64_t,4> sp(_sp);
            for (auto &v: sp)
                v &= (int64_t(1) << 40) - 1;
            auto inst = pulse(metadata, sp, cycles);
            if (tgt.type == Executor::PulseTarget::PLUT) {
                channels[chn].plut[tgt.addr] = inst;
            }
            else {
                pulses.push_back(inst);
            }
        }
        struct ChannelLUT {
            std::array<JaqalInst,4096> plut;
            std::array<uint16_t,4096> slut;
            std::array<std::pair<uint16_t,uint16_t>,4096> glut;
        };
        union {
            ChannelLUT channels[8];
            // std::memset only requires trivially-copyable
            // but GCC decides to warn as long as the types are non-trivial
            // (preventing us from defining default constructors)
            // Work around this requirement with a union char buffer.
            char _channel_mem[8 * sizeof(ChannelLUT)];
        };
    };

    struct ChannelGen {
        PulseAllocator pulses;
        std::vector<TimedID> pulse_ids;
        std::vector<int16_t> slut;
        std::vector<std::pair<int16_t,int16_t>> glut;
        std::vector<TimedID> gate_ids;

        void add_pulse(const JaqalInst &inst, int64_t cycle)
        {
            auto idx = pulses.get_addr(inst);
            if (idx >> PLUTW)
                throw std::length_error("Too many pulses in sequence.");
            pulse_ids.push_back({ .time = cycle, .id = int16_t(idx) });
        }

        void clear()
        {
            pulses.clear();
            pulse_ids.clear();
            slut.clear();
            glut.clear();
            gate_ids.clear();
        }

        int16_t add_glut(int16_t slut_addr1, int16_t slut_addr2)
        {
            auto gate_id = glut.size();
            if (gate_id >> GLUTW)
                throw std::length_error("Too many GLUT entries.");
            glut.push_back({ slut_addr1, slut_addr2 });
            return int16_t(gate_id);
        }
        int16_t add_slut(int npulses)
        {
            auto old_slut_len = slut.size();
            auto new_slut_len = old_slut_len + npulses;
            if ((new_slut_len - 1) >> SLUTW)
                throw std::length_error("Too many SLUT entries.");
            slut.resize(new_slut_len);
            return old_slut_len;
        }

        int16_t add_gate(std::span<TimedID> pulses)
        {
            auto npulses = (int)pulses.size();
            auto old_slut_len = add_slut(npulses);
            for (size_t i = 0; i < npulses; i++)
                slut[old_slut_len + i] = pulses[i].id;
            return add_glut(int16_t(old_slut_len), int16_t(old_slut_len + npulses - 1));
        }

        void sequence_gate(int16_t gid, int first_pid)
        {
            gate_ids.push_back({ .time = pulse_ids[first_pid].time, .id = gid });
        }

        void end()
        {
            // Keep the pulses that are added together together by using stable sort.
            // These currently belongs to the same tonedata
            // and there's a higher chance we have some reusable subsequences
            std::ranges::stable_sort(pulse_ids, [] (auto &a, auto &b) {
                return a.time < b.time;
            });
            int npulse = pulse_ids.size();
            if (!npulse)
                return;
            std::vector<int> pulse_str(npulse + 1);
            std::vector<int> pulse_sa(npulse + 1);
            std::vector<int> pulse_rk(npulse + 1);
            std::vector<int> pulse_height(npulse - 1);
            for (int i = 0; i < npulse; i++)
                pulse_str[i] = pulse_ids[i].id + 1;
            pulse_str[npulse] = 0;
            get_suffix_array(pulse_sa, pulse_str, pulse_rk);
            order_to_rank(pulse_rk, pulse_sa);
            for (int i = 0; i < npulse; i++)
                pulse_str[i] = pulse_ids[i].id + 1;
            pulse_str[npulse] = 0;
            get_height_array(pulse_height, pulse_str, pulse_sa, pulse_rk);

            struct SubStrInfoSA {
                int sa_begin;
            };
            struct SubStrInfoVec {
                std::vector<int> strs;
            };
            struct SubStrInfos {
                std::vector<SubStrInfoSA> sas;
                std::vector<SubStrInfoVec> vecs;
            };
            // Sort substrings according to repetition and length
            std::map<std::pair<int,int>,SubStrInfos> substrs;
            // Simple heuristic to avoid creating too many short gates
            foreach_max_range_min_val(pulse_height, 4, [&] (int i0, int i1, int str_len) {
                auto sa_begin = i0 + 1;
                auto nrep = i1 - i0 + 2;
                substrs[{nrep, str_len}].sas.push_back({ sa_begin });
            });

            // Map of start index -> (length, gate_id)
            std::map<int,std::pair<int,int>> substr_map;

            std::vector<int> substrs_cache;
            auto sort_substr = [&] (SubStrInfoSA sa, int nrep) {
                substrs_cache.clear();
                auto sa_begin = sa.sa_begin;
                auto sa_end = sa_begin + nrep;
                for (int sa_idx = sa_begin; sa_idx < sa_end; sa_idx++)
                    substrs_cache.push_back(pulse_sa[sa_idx]);
                std::ranges::sort(substrs_cache);
            };

            auto check_substr = [&] (std::vector<int> &substr_idxs, int str_len) {
                // Note that `substr_idxs` may alias `substrs_cache`.
                // but it doesn't alias anything else that's still in use elsewhere
                // so it can be freely mutated.

                assert(str_len > 1);
                int max_len = str_len;
                int nsubstrs = (int)substr_idxs.size();
                int last_substr = npulse;
                for (int i = nsubstrs - 1; i >= 0; i--) {
                    auto substr_idx = substr_idxs[i];
                    assert(substr_idx < last_substr);
                    int substr_len = std::min(max_len, last_substr - substr_idx);

                    auto it = substr_map.lower_bound(substr_idx);
                    if (it != substr_map.end()) {
                        assert(it->first >= substr_idx);
                        substr_len = std::min(substr_len, it->first - substr_idx);
                    }
                    if (it != substr_map.begin()) {
                        --it;
                        if (it->first + it->second.first > substr_idx) {
                            substr_len = 0;
                        }
                    }
                    assert(substr_len >= 0);
                    if (substr_len <= 1) {
                        substr_idxs.erase(substr_idxs.begin() + i);
                    }
                    else {
                        max_len = substr_len;
                        last_substr = substr_idx;
                    }
                }
                int new_nsubstrs = (int)substr_idxs.size();
                if (!new_nsubstrs)
                    return;
                if (new_nsubstrs < nsubstrs || max_len < str_len) {
                    assert(max_len > 1);
                    substrs[{new_nsubstrs, max_len}].vecs.emplace_back(
                        std::move(substr_idxs));
                    return;
                }
                assert(new_nsubstrs == nsubstrs && max_len == str_len);
                int gate_id = add_gate(std::span(pulse_ids)
                                       .subspan(substr_idxs[0], max_len));
                for (auto substr_idx: substr_idxs) {
                    auto [it, inserted] =
                        substr_map.insert({substr_idx, { max_len, gate_id }});
                    (void)it;
                    (void)inserted;
                    assert(inserted);
                }
            };

            while (!substrs.empty()) {
                auto it = --substrs.end();
                auto [nrep, str_len] = it->first;
                for (auto sa: it->second.sas) {
                    sort_substr(sa, nrep);
                    check_substr(substrs_cache, str_len);
                }
                for (auto &vec: it->second.vecs) {
                    check_substr(vec.strs, str_len);
                }
                substrs.erase(it);
            }
            bool has_single_pulse = false;
            int next_pulse = 0;
            for (auto [str_idx, substr_info]: substr_map) {
                assert(str_idx >= next_pulse);
                if (str_idx > next_pulse) {
                    int gate_id;
                    if (str_idx == next_pulse + 1) {
                        // Special de-dup handling for single pulse gate
                        has_single_pulse = true;
                        gate_id = -int(pulse_ids[next_pulse].id) - 1;
                    }
                    else {
                        gate_id = add_gate(std::span(pulse_ids)
                                           .subspan(next_pulse, str_idx - next_pulse));
                    }
                    sequence_gate(gate_id, next_pulse);
                }
                auto [max_len, gate_id] = substr_info;
                sequence_gate(gate_id, str_idx);
                next_pulse = str_idx + max_len;
            }
            if (npulse > next_pulse) {
                int gate_id = -1;
                if (npulse == next_pulse + 1) {
                    // Special de-dup handling for single pulse gate
                    has_single_pulse = true;
                    gate_id = -int(pulse_ids[next_pulse].id) - 1;
                }
                else {
                    gate_id = add_gate(std::span(pulse_ids)
                                       .subspan(next_pulse, npulse - next_pulse));
                }
                sequence_gate(gate_id, next_pulse);
            }
            if (has_single_pulse) {
                std::vector<int16_t> slut_map(1 << PLUTW, -1);
                assert((slut.size() >> 15) == 0);
                int16_t slut_len = (int16_t)slut.size();
                for (int16_t i = 0; i < slut_len; i++)
                    slut_map[slut[i]] = i;
                std::vector<int16_t> gate_map(1 << PLUTW, -1);
                auto get_single_gate_id = [&] (int16_t pid) {
                    assert((pid >> PLUTW) == 0);
                    int16_t gid = gate_map[pid];
                    if (gid < 0) {
                        int16_t slut_addr = slut_map[pid];
                        if (slut_addr < 0) {
                            slut_addr = add_slut(1);
                            slut[slut_addr] = pid;
                        }
                        gate_map[pid] = gid = add_glut(slut_addr, slut_addr);
                    }
                    return gid;
                };
                for (auto &gid: gate_ids) {
                    if (gid.id >= 0)
                        continue;
                    auto pid = -(gid.id + 1);
                    assert((pid >> 15) == 0);
                    gid.id = get_single_gate_id(int16_t(pid));
                }
            }
        }
    };

    static void print_inst(std::ostream &io, const JaqalInst &inst, bool print_float)
    {
        Printer printer{io, print_float};
        Executor::execute(printer, inst);
    }

    static void print_insts(std::ostream &io, const char *p, size_t sz, bool print_float)
    {
        Printer printer{io, print_float};
        Executor::execute(printer, std::span(p, sz));
    }

    static auto extract_pulses(const char *p, size_t sz)
    {
        PulseSequencer sequencer;
        Executor::execute(sequencer, std::span(p, sz));
        return sequencer.pulses;
    }

    __attribute__((returns_nonnull))
    static PyObject *inst_to_dict(const JaqalInst &inst)
    {
        DictConverter converter;
        Executor::execute(converter, inst);
        return converter.dict.release();
    }

    // Set the minimum clock cycles for a pulse to help avoid underflows. This time
    // is determined by state machine transitions for loading another gate, but does
    // not account for serialization of pulse words.
    static constexpr int MINIMUM_PULSE_CLOCK_CYCLES = 4;
};

struct Jaqal_v1_3 {
    // All the instructions are 256 bits (32 bytes) and there are 7 types of
    // instructions roughly divided into 2 groups, programming and sequence output.
    // The two groups are determined by the GSEQ_ENABLE (247) bit,
    // where bit value 0 means programming mode and bit value 1 means
    // sequence output mode.
    // For either mode, the next two bits (PROG_MODE / SEQ_MODE) indicates
    // the exact type of the instruction and the meaning of the values is defined
    // in ProgMode and SeqMode respectively.
    enum class ProgMode: uint8_t {
        // 0 not used
        GLUT = 1, // Gate LUT
        SLUT = 2, // Sequence LUT
        PLUT = 3, // Pulse LUT
    };
    enum class SeqMode: uint8_t {
        // Sequence gate IDs
        GATE = 0,
        // Wait for Ancilla trigger, and sequence gate IDs based on readout
        WAIT_ANC = 1,
        // Continue sequencing based on previous ancilla data
        CONT_ANC = 2,
        // Bypass LUT/streaming mode
        STREAM = 3,
    };

    // Other than these bits, the only other bits thats shared by all types
    // of instructions appears to be the channel number bits [DMA_MUX+7, DMA_MUX].
    // The programming, sequencing as well as the lookup table storage appears to be
    // completely and statically segregated for each channels.

    // Both the STREAM instruction and the Pulse LUT instruction are used to
    // set the low level pulse/spline parameters and therefore shares very similar
    // format, i.e. the raw data format. The instruction consists of 6 parts,
    // |255       200|199       160|159 120|119  80|79   40|39    0|
    // |Metadata (56)|duration (40)|U3 (40)|U2 (40)|U1 (40)|U0 (40)|
    // where the duration and the U0-U3 fields specify the output parameters,
    // for a output parameter. The metadata part (bits [255, 200])
    // includes the shared bits mentioned above as well as other bits that
    // are meant for the spline engine/DDS.

    // The metadata format also depends on the type of parameters,
    // which is determined by the MODTYPE bitfield (bits [223, 216]) and the bit names
    // are defined in ModType.
    enum ModType: uint8_t {
        FRQMOD0 = 0,
        AMPMOD0 = 1,
        PHSMOD0 = 2,
        FRQMOD1 = 3,
        AMPMOD1 = 4,
        PHSMOD1 = 5,
        FRMROT0 = 6,
        FRMROT1 = 7,
    };
    enum ModTypeMask: uint8_t {
        FRQMOD0_MASK = 1 << FRQMOD0,
        AMPMOD0_MASK = 1 << AMPMOD0,
        PHSMOD0_MASK = 1 << PHSMOD0,
        FRQMOD1_MASK = 1 << FRQMOD1,
        AMPMOD1_MASK = 1 << AMPMOD1,
        PHSMOD1_MASK = 1 << PHSMOD1,
        FRMROT0_MASK = 1 << FRMROT0,
        FRMROT1_MASK = 1 << FRMROT1,
    };
    // For the frequency/amplitude/phase parameters, the metadata format was
    // |255 248|247              245|244 241|240   229|228 224|223 216|215 213|
    // |DMA_MUX|GSEQ_ENABLE+SEQ_MODE|   0   |PLUT_ADDR|   0   |MODTYPE|   0   |
    // |212  208|   207   |   206   |   205   |   204   |   203   |202 200|
    // |SPLSHIFT|AMP_FB_EN|FRQ_FB_EN|OUTPUT_EN|SYNC_FLAG|WAIT_TRIG|   0   |

    // For the frame rotation parameter, the metadata format was
    // |255 248|247              245|244 241|240   229|228 224|223 216|215 213|
    // |DMA_MUX|GSEQ_ENABLE+SEQ_MODE|   0   |PLUT_ADDR|   0   |MODTYPE|   0   |
    // |212  208|   207   |   206   |    205   |    204   |   203   |
    // |SPLSHIFT|APPLY_EOF|CLR_FRAME|FWD_FRM_T1|FWD_FRM_T0|WAIT_TRIG|
    // |    202   |    201   |200|
    // |INV_FRM_T1|INV_FRM_T0| 0 |

    // The AMP_FB_EN, FRQ_FB_EN, OUTPUT_FB_EN, SYNC_FLAG, APPLY_EOF, CLR_FRAME,
    // FWD_FRM_T1, FWD_FRM_T0, INV_FRM_T1, INV_FRM_T0 are flags that affects
    // the behavior of the specific spline engines/DDS and are documented below.

    // SPLSHIFT is the shift that should be applied on the higher order terms
    // in the spline coefficient.
    // PLUT_ADDR is only used for the PLUT programming instruction and is the
    // 12 bit address to locate the pulse in the PLUT.
    // WAIT_TRIG is a flag to decide whether this command should wait for the trigger
    // to happen before being consumed by the spline engine.

    // The format for the SLUT programming instruction is,
    // |255 248|247              245|244 239|238  236|235 198|
    // |DMA_MUX|GSEQ_ENABLE+SEQ_MODE|   0   |SLUT_CNT|   0   |
    // |197  185|184    177|176  165|164  152|151    144|143  132|
    // | SADDR5 | MODTYPE5 | PADDR5 | SADDR4 | MODTYPE4 | PADDR4 |
    // |131  119|118    111|110   99|98    86|85      78|77    66|
    // | SADDR3 | MODTYPE3 | PADDR3 | SADDR2 | MODTYPE2 | PADDR2 |
    // |65    53|52      45|44    33|32    20|19      12|11     0|
    // | SADDR1 | MODTYPE1 | PADDR1 | SADDR0 | MODTYPE0 | PADDR0 |

    // The format for the GLUT programming instruction is,
    // |255 248|247              245|244 239|238  236|235 228|
    // |DMA_MUX|GSEQ_ENABLE+SEQ_MODE|   0   |GLUT_CNT|   0   |
    // |227  216|215  203|202 190|189  178|177  165|164 152|
    // | GADDR5 | START5 |  END5 | GADDR4 | START4 |  END4 |
    // |151  140|139  127|126 114|113  102|101   89|88  76|
    // | GADDR3 | START3 |  END3 | GADDR2 | START2 | END2 |
    // |75    64|63    51|50  38|37    26|25    13|12   0|
    // | GADDR1 | START1 | END1 | GADDR0 | START0 | END0 |

    // The format for the gate sequence instruction is,
    // |255 248|247              245|244|243  239|238 220|
    // |DMA_MUX|GSEQ_ENABLE+SEQ_MODE| 0 |GSEQ_CNT|   0   |
    // |219   209|208   198|197   187|186   176|175   165|164   154|
    // | GADDR19 | GADDR18 | GADDR17 | GADDR16 | GADDR15 | GADDR14 |
    // |153   143|142   132|131   121|120   110|109   99|98    88|87    77|
    // | GADDR13 | GADDR12 | GADDR11 | GADDR10 | GADDR9 | GADDR8 | GADDR7 |
    // |76    66|65    55|54    44|43    33|32    22|21    11|10     0|
    // | GADDR6 | GADDR5 | GADDR4 | GADDR3 | GADDR2 | GADDR1 | GADDR0 |

    // For all three formats, bit 0 to bit 228 (really, bit 227 due to the size of
    // the element) are packed with multiple elements of programming/sequence word
    // (13 + 20 bits for SLUT programming, 12 + 13 * 2 bits for GLUT programming
    // and 11 bits for gate sequence). The number of elements packed
    // in the instruction is recorded in a count field (LSB at bit 236 for programming
    // and LSB at bit 239 for sequence instruction).

    struct Bits {
        static constexpr int DMA_MUX = 248;
        static constexpr int GSEQ_ENABLE = 247;
        static constexpr int PROG_MODE = 245;
        static constexpr int SEQ_MODE = 245;

        // Number of packed GLUT programming words
        static constexpr int GLUT_CNT = 236;
        // Number of packed SLUT programming words
        static constexpr int SLUT_CNT = 236;
        // Number of packed gate sequence identifiers
        static constexpr int GSEQ_CNT = 239;

        static constexpr int PLUT_ADDR = 229;
        static constexpr int PACKING_LIMIT = 229;

        // Modulation type (freq/phase/amp/framerot)
        static constexpr int MODTYPE = 216;
        // Fixed point shift for spline coefficients
        static constexpr int SPLSHIFT = 208;

        //// For normal parameters
        // Amplitude feedback enable (placeholder)
        static constexpr int AMP_FB_EN = 207;
        // Frequency feedback enable
        static constexpr int FRQ_FB_EN = 206;
        // Toggle output enable
        static constexpr int OUTPUT_EN = 205;
        // Apply global synchronization
        static constexpr int SYNC_FLAG = 204;

        //// For frame rotation parameters
        // Apply frame rotation at end of pulse
        static constexpr int APPLY_EOF = 207;
        // Clear frame accumulator
        static constexpr int CLR_FRAME = 206;
        // Forward frame to tone 1/0
        static constexpr int FWD_FRM = 204;

        // Wait for external trigger
        static constexpr int WAIT_TRIG = 203;

        //// Additional bits for frame rotation parameters
        // Invert sign on frame for tone 1/0
        static constexpr int INV_FRM = 201;

        // Start of metadata for raw data instruction
        static constexpr int METADATA = 200;
    };

    static constexpr inline auto
    pulse(uint64_t metadata, const std::array<int64_t,4> &isp, int64_t cycles)
    {
        assert((isp[0] >> 40) == 0);
        assert((isp[1] >> 40) == 0);
        assert((isp[2] >> 40) == 0);
        assert((isp[3] >> 40) == 0);
        assert((cycles >> 40) == 0);
        assume((isp[0] >> 40) == 0);
        assume((isp[1] >> 40) == 0);
        assume((isp[2] >> 40) == 0);
        assume((isp[3] >> 40) == 0);
        assume((cycles >> 40) == 0);

        std::array<int64_t,4> data{
            isp[0] | (isp[1] << 40),
            (isp[1] >> (64 - 40)) | (isp[2] << (80 - 64)) | (isp[3] << (120 - 64)),
            (isp[3] >> (128 - 120)) | (cycles << (160 - 128)),
            (cycles >> (192 - 160)) | int64_t(metadata << (200 - 192)),
        };
        return JaqalInst(data);
    }

    static constexpr auto modtype_mask =
        JaqalInst::mask(Bits::MODTYPE, Bits::MODTYPE + 7);
    static constexpr auto channel_mask =
        JaqalInst::mask(Bits::DMA_MUX, Bits::DMA_MUX + 7);
    static constexpr auto modtype_nmask = ~modtype_mask;
    static constexpr auto channel_nmask = ~channel_mask;
    static constexpr inline auto
    apply_modtype_mask(JaqalInst pulse, ModTypeMask mod_mask)
    {
        return (pulse & modtype_nmask) | JaqalInst(uint8_t(mod_mask)) << Bits::MODTYPE;
    }
    static constexpr inline auto
    apply_channel_mask(JaqalInst pulse, uint8_t chn_mask)
    {
        return (pulse & channel_nmask) | JaqalInst(chn_mask) << Bits::DMA_MUX;
    }

    static constexpr inline uint64_t raw_param_metadata(
        int shift_len, bool waittrig, bool sync, bool enable, bool fb_enable)
    {
        assert(shift_len >= 0 && shift_len < 32);
        uint64_t metadata = uint64_t(shift_len) << (Bits::SPLSHIFT - Bits::METADATA);
        metadata |= uint64_t(waittrig) << (Bits::WAIT_TRIG - Bits::METADATA);
        metadata |= uint64_t(enable) << (Bits::OUTPUT_EN - Bits::METADATA);
        metadata |= uint64_t(fb_enable) << (Bits::FRQ_FB_EN - Bits::METADATA);
        metadata |= uint64_t(sync) << (Bits::SYNC_FLAG - Bits::METADATA);
        return metadata;
    }
    static constexpr inline auto
    freq_pulse(cubic_spline_t sp, int64_t cycles, bool waittrig, bool sync, bool fb_enable)
    {
        assert(cycles >= 4);
        assert((cycles >> 40) == 0);
        auto [isp, shift_len] = convert_pdq_spline_freq(sp.to_array(), cycles);
        auto metadata = raw_param_metadata(shift_len, waittrig, sync, false, fb_enable);
        return pulse(metadata, isp, cycles);
    }
    static constexpr inline auto
    amp_pulse(cubic_spline_t sp, int64_t cycles, bool waittrig,
              bool sync=false, bool fb_enable=false)
    {
        assert(cycles >= 4);
        assert((cycles >> 40) == 0);
        auto [isp, shift_len] = convert_pdq_spline_amp(sp.to_array(), cycles);
        auto metadata = raw_param_metadata(shift_len, waittrig, sync, false, fb_enable);
        return pulse(metadata, isp, cycles);
    }
    static constexpr inline auto
    phase_pulse(cubic_spline_t sp, int64_t cycles, bool waittrig,
                bool sync=false, bool fb_enable=false)
    {
        assert(cycles >= 4);
        assert((cycles >> 40) == 0);
        auto [isp, shift_len] = convert_pdq_spline_phase(sp.to_array(), cycles);
        auto metadata = raw_param_metadata(shift_len, waittrig, sync, false, fb_enable);
        return pulse(metadata, isp, cycles);
    }

    static constexpr inline uint64_t raw_frame_metadata(
        int shift_len, bool waittrig, bool apply_at_end, bool rst_frame,
        int fwd_frame_mask, int inv_frame_mask)
    {
        assert(shift_len >= 0 && shift_len < 32);
        uint64_t metadata = uint64_t(shift_len) << (Bits::SPLSHIFT - Bits::METADATA);
        metadata |= uint64_t(waittrig) << (Bits::WAIT_TRIG - Bits::METADATA);
        metadata |= uint64_t(apply_at_end) << (Bits::APPLY_EOF - Bits::METADATA);
        metadata |= uint64_t(rst_frame) << (Bits::CLR_FRAME - Bits::METADATA);
        metadata |= uint64_t(fwd_frame_mask) << (Bits::FWD_FRM - Bits::METADATA);
        metadata |= uint64_t(inv_frame_mask) << (Bits::INV_FRM - Bits::METADATA);
        return metadata;
    }
    static constexpr inline auto
    frame_pulse(cubic_spline_t sp, int64_t cycles, bool waittrig, bool apply_at_end,
                bool rst_frame, int fwd_frame_mask, int inv_frame_mask)
    {
        assert(cycles >= 4);
        assert((cycles >> 40) == 0);
        auto [isp, shift_len] = convert_pdq_spline_phase(sp.to_array(), cycles);
        auto metadata = raw_frame_metadata(shift_len, waittrig, apply_at_end,
                                           rst_frame, fwd_frame_mask, inv_frame_mask);
        return pulse(metadata, isp, cycles);
    }

    // LUT Address Widths
    // These values contain the number of bits used for an address for a
    // particular LUT. The address width is the same for reading and writing
    // data for the Pulse LUT (PLUT) and the Sequence LUT (SLUT), but the
    // address width is asymmetric for the Gate LUT (GLUT). This is because
    // the read address is partly completed by external hardware inputs and
    // the read address size (GLUTW) is thus smaller than the write address
    // size (GPRGW) used to program the GLUT
    static constexpr int GPRGW = 12;  // Gate LUT write address width
    static constexpr int GLUTW = 11;  // Gate LUT read address width
    static constexpr int SLUTW = 13;  // Sequence LUT address width
    // 12 bit plut address + 8bit parameter mask
    static constexpr int SLUTDW = 20;  // Sequence LUT data width
    static constexpr int PLUTW = 12;  // Pulse LUT address width

    static constexpr int SLUT_ELSZ = SLUTW + SLUTDW;
    static constexpr int GLUT_ELSZ = GPRGW + 2 * SLUTW;
    static constexpr int GSEQ_ELSZ = GLUTW;
    // Number of programming or gate sequence words that can be packed into a single
    // transfer. PLUT programming data is always one word per transfer.
    static constexpr int SLUT_MAXCNT = Bits::PACKING_LIMIT / SLUT_ELSZ;
    static constexpr int GLUT_MAXCNT = Bits::PACKING_LIMIT / GLUT_ELSZ;
    static constexpr int GSEQ_MAXCNT = Bits::PACKING_LIMIT / GSEQ_ELSZ;

    static constexpr inline JaqalInst stream(JaqalInst pulse)
    {
        pulse |= JaqalInst(uint8_t(SeqMode::STREAM)) << Bits::SEQ_MODE;
        pulse |= JaqalInst(1) << Bits::GSEQ_ENABLE;
        return pulse;
    }

    static constexpr inline JaqalInst program_PLUT(JaqalInst pulse, uint16_t addr)
    {
        assert((addr >> PLUTW) == 0);
        pulse |= JaqalInst(uint8_t(ProgMode::PLUT)) << Bits::PROG_MODE;
        pulse |= JaqalInst(addr) << Bits::PLUT_ADDR;
        return pulse;
    }

    static constexpr inline auto
    program_SLUT(uint8_t chn_mask, const uint16_t *saddrs,
                 const ModTypeMask *mod_types, const uint16_t *paddrs, int n)
    {
        JaqalInst inst;
        assert(n <= SLUT_MAXCNT);
        for (int i = 0; i < n; i++) {
            assert((paddrs[i] >> PLUTW) == 0);
            inst |= JaqalInst(paddrs[i]) << SLUT_ELSZ * i;
            inst |= JaqalInst(uint8_t(mod_types[i])) << (SLUT_ELSZ * i + PLUTW);
            assert((saddrs[i] >> SLUTW) == 0);
            inst |= JaqalInst(saddrs[i]) << (SLUT_ELSZ * i + SLUTDW);
        }
        inst |= JaqalInst(uint8_t(ProgMode::SLUT)) << Bits::PROG_MODE;
        inst |= JaqalInst(uint8_t(n)) << Bits::SLUT_CNT;
        inst |= JaqalInst(chn_mask) << Bits::DMA_MUX;
        return inst;
    }

    static constexpr inline auto
    program_GLUT(uint8_t chn_mask, const uint16_t *gaddrs, const uint16_t *starts,
                 const uint16_t *ends, int n)
    {
        JaqalInst inst;
        assert(n <= GLUT_MAXCNT);
        for (int i = 0; i < n; i++) {
            assert((gaddrs[i] >> GPRGW) == 0);
            inst |= JaqalInst(gaddrs[i]) << (GLUT_ELSZ * i + SLUTW * 2);
            assert((ends[i] >> SLUTW) == 0);
            inst |= JaqalInst(ends[i]) << (GLUT_ELSZ * i + SLUTW);
            assert((starts[i] >> SLUTW) == 0);
            inst |= JaqalInst(starts[i]) << (GLUT_ELSZ * i);
        }
        inst |= JaqalInst(uint8_t(ProgMode::GLUT)) << Bits::PROG_MODE;
        inst |= JaqalInst(uint8_t(n)) << Bits::GLUT_CNT;
        inst |= JaqalInst(chn_mask) << Bits::DMA_MUX;
        return inst;
    }

    static constexpr inline auto
    sequence(uint8_t chn_mask, SeqMode m, uint16_t *gaddrs, int n)
    {
        JaqalInst inst;
        assert(n <= GSEQ_MAXCNT);
        assert(m != SeqMode::STREAM);
        for (int i = 0; i < n; i++) {
            assert((gaddrs[i] >> GLUTW) == 0);
            inst |= JaqalInst(gaddrs[i]) << (GSEQ_ELSZ * i);
        }
        inst |= JaqalInst(uint8_t(m)) << Bits::SEQ_MODE;
        inst |= JaqalInst(1) << Bits::GSEQ_ENABLE;
        inst |= JaqalInst(n) << Bits::GSEQ_CNT;
        inst |= JaqalInst(chn_mask) << Bits::DMA_MUX;
        return inst;
    }

    static constexpr inline uint8_t get_chn_mask(const JaqalInst &inst)
    {
        return uint8_t((inst >> Bits::DMA_MUX)[0]);
    }

    struct Executor {
        enum class Error {
            Reserved,
            GLUT_OOB,
            SLUT_OOB,
            GSEQ_OOB,
        };
        static const char *error_msg(Error err)
        {
            switch (err) {
            default:
            case Error::Reserved:
                return "reserved";
            case Error::GLUT_OOB:
                return "glut_oob";
            case Error::SLUT_OOB:
                return "slut_oob";
            case Error::GSEQ_OOB:
                return "gseq_oob";
            }
        }
        struct PulseTarget {
            enum Type {
                None,
                PLUT,
                Stream,
            } type;
            uint16_t addr;
        };
        struct PulseMeta {
            bool trig;
            bool fb;
            bool en;
            bool sync;
            bool apply_eof;
            bool clr_frame;
            int8_t fwd_frame_mask;
            int8_t inv_frame_mask;
        };
        static cubic_spline_t freq_spline(PDQSpline spl, uint64_t cycles)
        {
            spl.scale = output_clock / double(1ll << 40);
            return spl.get_spline(cycles);
        }
        static cubic_spline_t amp_spline(PDQSpline spl, uint64_t cycles)
        {
            spl.scale = 1 / double(((1ll << 16) - 1ll) << 23);
            return spl.get_spline(cycles);
        }
        static cubic_spline_t phase_spline(PDQSpline spl, uint64_t cycles)
        {
            spl.scale = 1 / double(1ll << 40);
            return spl.get_spline(cycles);
        }
        template<typename T>
        static void execute(auto &&cb, const std::span<T> insts, bool allow_op0=false)
        {
            auto sz = insts.size_bytes();
            if (sz % sizeof(JaqalInst) != 0)
                throw std::invalid_argument("Instruction stream length "
                                            "not a multiple of instruction size");
            auto p = (const char*)insts.data();
            for (size_t i = 0; i < sz; i += sizeof(JaqalInst)) {
                JaqalInst inst;
                memcpy(&inst, p + i, sizeof(JaqalInst));
                if (i != 0)
                    cb.next();
                execute(cb, inst, allow_op0);
            }
        }
        static void execute(auto &&cb, const JaqalInst &inst, bool allow_op0=true)
        {
            int op = (inst >> Bits::PROG_MODE)[0] & 0x7;
            switch (op) {
            default:
            case 0:
                if (!allow_op0) {
                    invalid(cb, inst, Error::Reserved);
                    return;
                }
                pulse(cb, inst, PulseTarget::None);
                return;
            case int(ProgMode::PLUT):
                pulse(cb, inst, PulseTarget::PLUT);
                return;
            case int(SeqMode::STREAM) | 4:
                pulse(cb, inst, PulseTarget::Stream);
                return;

            case int(ProgMode::GLUT):
                GLUT(cb, inst);
                return;
            case int(ProgMode::SLUT):
                SLUT(cb, inst);
                return;
            case int(SeqMode::GATE) | 4:
            case int(SeqMode::WAIT_ANC) | 4:
            case int(SeqMode::CONT_ANC) | 4:
                GSEQ(cb, inst, SeqMode(op & 3));
                return;
            }
        }
    private:
        static void invalid(auto &&cb, const JaqalInst &inst, Error err)
        {
            cb.invalid(inst, err);
        }
        static void GLUT(auto &&cb, const JaqalInst &inst)
        {
            if (test_bits(inst, 239, 244) || test_bits(inst, 228, 235)) {
                invalid(cb, inst, Error::Reserved);
                return;
            }
            int cnt = (inst >> Bits::GLUT_CNT)[0] & 0x7;
            uint8_t chn_mask = get_chn_mask(inst);
            if (cnt > GLUT_MAXCNT) {
                invalid(cb, inst, Error::GLUT_OOB);
                return;
            }
            else if (test_bits(inst, GLUT_ELSZ * cnt, 227)) {
                invalid(cb, inst, Error::Reserved);
                return;
            }
            uint16_t gaddrs[GLUT_MAXCNT];
            uint16_t starts[GLUT_MAXCNT];
            uint16_t ends[GLUT_MAXCNT];
            for (int i = 0; i < cnt; i++) {
                auto w = (inst >> GLUT_ELSZ * i)[0];
                gaddrs[i] = (w >> (SLUTW * 2)) & ((1 << GPRGW) - 1);
                ends[i] = (w >> SLUTW) & ((1 << SLUTW) - 1);
                starts[i] = w & ((1 << SLUTW) - 1);
            }
            cb.GLUT(chn_mask, gaddrs, starts, ends, cnt);
        }

        static void SLUT(auto &&cb, const JaqalInst &inst)
        {
            if (test_bits(inst, 239, 244) || test_bits(inst, 198, 235)) {
                invalid(cb, inst, Error::Reserved);
                return;
            }
            int cnt = (inst >> Bits::SLUT_CNT)[0] & 0xf;
            uint8_t chn_mask = get_chn_mask(inst);
            if (cnt > SLUT_MAXCNT) {
                invalid(cb, inst, Error::SLUT_OOB);
                return;
            }
            else if (test_bits(inst, SLUT_ELSZ * cnt, 219)) {
                invalid(cb, inst, Error::Reserved);
                return;
            }
            uint16_t saddrs[SLUT_MAXCNT];
            ModTypeMask mod_types[SLUT_MAXCNT];
            uint16_t paddrs[SLUT_MAXCNT];
            for (int i = 0; i < cnt; i++) {
                auto w = (inst >> SLUT_ELSZ * i)[0];
                saddrs[i] = (w >> SLUTDW) & ((1 << SLUTW) - 1);
                mod_types[i] = ModTypeMask((w >> PLUTW) & 0xff);
                paddrs[i] = w & ((1 << PLUTW) - 1);
            }
            cb.SLUT(chn_mask, saddrs, mod_types, paddrs, cnt);
        }

        static void GSEQ(auto &&cb, const JaqalInst &inst, SeqMode m)
        {
            if (test_bits(inst, 244, 244) || test_bits(inst, 220, 238)) {
                invalid(cb, inst, Error::Reserved);
                return;
            }
            int cnt = (inst >> Bits::GSEQ_CNT)[0] & 0x1f;
            uint8_t chn_mask = get_chn_mask(inst);
            if (cnt > GSEQ_MAXCNT) {
                invalid(cb, inst, Error::GSEQ_OOB);
                return;
            }
            else if (test_bits(inst, GSEQ_ELSZ * cnt, 227)) {
                invalid(cb, inst, Error::Reserved);
                return;
            }
            uint16_t gaddrs[GSEQ_MAXCNT];
            for (int i = 0; i < cnt; i++) {
                auto w = (inst >> GSEQ_ELSZ * i)[0];
                gaddrs[i] = w & ((1 << GLUTW) - 1);
            }
            cb.GSEQ(chn_mask, gaddrs, cnt, m);
        }

        static void pulse(auto &&cb, const JaqalInst &inst, PulseTarget::Type type)
        {
            uint16_t addr = (inst >> Bits::PLUT_ADDR)[0] & ((1 << PLUTW) - 1);
            if (type != PulseTarget::PLUT && addr) {
                invalid(cb, inst, Error::Reserved);
                return;
            }
            PulseTarget tgt{type, addr};
            if (test_bits(inst, 241, 244) || test_bits(inst, 224, 228) ||
                test_bits(inst, 213, 215)) {
                invalid(cb, inst, Error::Reserved);
                return;
            }
            uint8_t chn_mask = get_chn_mask(inst);
            PDQSpline spl;
            spl.shift = (inst >> Bits::SPLSHIFT)[0] & 0x1f;
            auto load_s40 = [&] (int offset) {
                auto mask = (int64_t(1) << 40) - 1;
                auto data = (inst >> offset)[0] & mask;
                if (data & (int64_t(1) << 39))
                    return data | ~mask;
                return data;
            };
            spl.orders[0] = load_s40(40 * 0);
            spl.orders[1] = load_s40(40 * 1);
            spl.orders[2] = load_s40(40 * 2);
            spl.orders[3] = load_s40(40 * 3);
            int64_t cycles = (inst >> 40 * 4)[0] & ((int64_t(1) << 40) - 1);
            auto mod_type = ModTypeMask((inst >> Bits::MODTYPE)[0] & 0xff);
            if (mod_type & (FRMROT0_MASK | FRMROT1_MASK) ||
                tgt.type == Executor::PulseTarget::PLUT) {
                if (test_bits(inst, 200, 200)) {
                    invalid(cb, inst, Error::Reserved);
                    return;
                }
            }
            else if (test_bits(inst, 200, 202) ||
                     test_bits(inst, Bits::AMP_FB_EN, Bits::AMP_FB_EN)) {
                invalid(cb, inst, Error::Reserved);
                return;
            }
            PulseMeta meta{
                .trig = bool((inst >> Bits::WAIT_TRIG)[0] & 1),
                .fb = bool((inst >> Bits::FRQ_FB_EN)[0] & 1),
                .en = bool((inst >> Bits::OUTPUT_EN)[0] & 1),
                .sync = bool((inst >> Bits::SYNC_FLAG)[0] & 1),
                .apply_eof = bool((inst >> Bits::APPLY_EOF)[0] & 1),
                .clr_frame = bool((inst >> Bits::CLR_FRAME)[0] & 1),
                .fwd_frame_mask = int8_t((inst >> Bits::FWD_FRM)[0] & 3),
                .inv_frame_mask = int8_t((inst >> Bits::INV_FRM)[0] & 3),
            };
            cb.pulse(chn_mask, mod_type, spl, cycles, meta, tgt);
        }
    };

    struct Printer {
        void invalid(const JaqalInst &inst, Executor::Error err)
        {
            io << "invalid(" << Executor::error_msg(err) << "): "
               << std::noshowbase << inst;
        }

        void next()
        {
            io << std::endl;
        }

        void GLUT(uint8_t chn_mask, const uint16_t *gaddrs, const uint16_t *starts,
                  const uint16_t *ends, int cnt)
        {
            io << "glut";
            print_chn_mask(chn_mask);
            io << "[" << cnt << "]";
            for (int i = 0; i < cnt; i++) {
                io << " [" << gaddrs[i] << "]=[" << starts[i] << "," << ends[i] << "]";
            }
        }
        void SLUT(uint8_t chn_mask, const uint16_t *saddrs,
                  const ModTypeMask *mod_types, const uint16_t *paddrs, int cnt)
        {
            io << "slut";
            print_chn_mask(chn_mask);
            io << "[" << cnt << "]";
            for (int i = 0; i < cnt; i++) {
                io << " [" << saddrs[i] << "]=" << paddrs[i];
                print_mod_type(mod_types[i], true);
            }
        }
        void GSEQ(uint8_t chn_mask, const uint16_t *gaddrs, int cnt, SeqMode m)
        {
            if (m == SeqMode::GATE) {
                io << "gseq";
            }
            else if (m == SeqMode::WAIT_ANC) {
                io << "wait_anc";
            }
            else if (m == SeqMode::CONT_ANC) {
                io << "cont_anc";
            }
            print_chn_mask(chn_mask);
            io << "[" << cnt << "]";
            for (int i = 0; i < cnt; i++) {
                io << " " << gaddrs[i];
            }
        }

        void pulse(int chn_mask, ModTypeMask mod_type, const PDQSpline &spl,
                   int64_t cycles, Executor::PulseMeta meta, Executor::PulseTarget tgt)
        {
            if (tgt.type == Executor::PulseTarget::None) {
                io << "pulse_data";
            }
            else if (tgt.type == Executor::PulseTarget::PLUT) {
                io << "plut";
            }
            else if (tgt.type == Executor::PulseTarget::Stream) {
                io << "stream";
            }
            print_chn_mask(chn_mask);
            io << " ";
            if (tgt.type == Executor::PulseTarget::PLUT)
                io << "[" << tgt.addr << "]={}";
            else
                print_mod_type(mod_type, false);
            io << " <" << cycles << ">";
            auto print_orders = [&] (auto order1, auto order2, auto order3, auto cb) {
                if (order1 || order2 || order3)
                    cb(1, order1);
                if (order2 || order3)
                    cb(2, order2);
                if (order3)
                    cb(3, order3);
            };
            if (print_float && tgt.type != Executor::PulseTarget::PLUT && mod_type) {
                bool freq = mod_type & (FRQMOD0_MASK | FRQMOD1_MASK);
                bool amp = mod_type & (AMPMOD0_MASK | AMPMOD1_MASK);
                bool phase = mod_type & (PHSMOD0_MASK | PHSMOD1_MASK |
                                         FRMROT0_MASK | FRMROT1_MASK);
                bool print_name = (int(freq) + int(amp) + int(phase)) > 1;
                auto print_spl = [&] (const char *name, auto &&cb, bool cond) {
                    if (!cond)
                        return;
                    auto cspl = cb(spl, cycles);
                    io << " ";
                    if (print_name)
                        io << name;
                    io << "{";
                    format_double(io, cspl.order0);
                    print_orders(cspl.order1, cspl.order2, cspl.order3,
                                 [&] (int, auto order) {
                                     format_double(io << ", ", order); });
                    io << "}";
                };
                print_spl("freq", Executor::freq_spline, freq);
                print_spl("amp", Executor::amp_spline, amp);
                print_spl("phase", Executor::phase_spline, phase);
            }
            else {
                io << " {";
                io << std::showbase << std::hex << spl.orders[0];
                print_orders(spl.orders[1], spl.orders[2], spl.orders[3],
                             [&] (int i, auto order) {
                                 io << ", " << std::showbase << std::hex << order;
                                 if (spl.shift) {
                                     io << ">>" << std::dec << (spl.shift * i);
                                 }
                             });
                io << "}";
            }
            auto print_flag = [&] (const char *name, bool cond) {
                if (cond) {
                    io << " " << name;
                }
            };
            print_flag("trig", meta.trig);
            if (mod_type & (FRQMOD0_MASK | AMPMOD0_MASK | PHSMOD0_MASK |
                            FRQMOD1_MASK | AMPMOD1_MASK | PHSMOD1_MASK) ||
                tgt.type == Executor::PulseTarget::PLUT) {
                print_flag("sync", meta.sync);
                print_flag("enable", meta.en);
                print_flag("ff", meta.fb);
            }
            if (mod_type & (FRMROT0_MASK | FRMROT1_MASK) ||
                tgt.type == Executor::PulseTarget::PLUT) {
                print_flag("eof", meta.apply_eof);
                print_flag("clr", meta.clr_frame);
                io << " fwd:" << std::dec << std::noshowbase << int(meta.fwd_frame_mask)
                   << " inv:" << std::dec << std::noshowbase << int(meta.inv_frame_mask);
            }
        }

        std::ostream &io;
        bool print_float{true};
    private:
        void print_list_ele(const char *name, bool cond, bool &first)
        {
            if (!cond)
                return;
            if (!std::exchange(first, false))
                io << ",";
            io << name;
        }
        void print_mod_type(ModTypeMask mod_type, bool force_brace)
        {
            bool use_brace = force_brace || (std::popcount(uint8_t(mod_type)) != 1);
            bool first = true;
            if (use_brace) {
                io << "{";
            }
            print_list_ele("freq0", mod_type & FRQMOD0_MASK, first);
            print_list_ele("amp0", mod_type & AMPMOD0_MASK, first);
            print_list_ele("phase0", mod_type & PHSMOD0_MASK, first);
            print_list_ele("freq1", mod_type & FRQMOD1_MASK, first);
            print_list_ele("amp1", mod_type & AMPMOD1_MASK, first);
            print_list_ele("phase1", mod_type & PHSMOD1_MASK, first);
            print_list_ele("frame_rot0", mod_type & FRMROT0_MASK, first);
            print_list_ele("frame_rot1", mod_type & FRMROT1_MASK, first);
            if (use_brace) {
                io << "}";
            }
        }
        void print_chn_mask(uint8_t chn_mask)
        {
            if (chn_mask == 0xff) {
                io << ".all";
            }
            else {
                bool single = std::popcount(chn_mask) == 1;
                bool first = true;
                io << (single ? "." : "{");
                print_list_ele("0", chn_mask & 1, first);
                print_list_ele("1", chn_mask & 2, first);
                print_list_ele("2", chn_mask & 4, first);
                print_list_ele("3", chn_mask & 8, first);
                print_list_ele("4", chn_mask & 16, first);
                print_list_ele("5", chn_mask & 32, first);
                print_list_ele("6", chn_mask & 64, first);
                print_list_ele("7", chn_mask & 128, first);
                if (!single) {
                    io << "}";
                }
            }
        }
    };

    struct DictConverter {
        void invalid(const JaqalInst &inst, Executor::Error err)
        {
            throw_if(PyDict_SetItem(dict, type_str(), invalid_str()));
            py_object msg(pyunicode_from_string(Executor::error_msg(err)));
            throw_if(PyDict_SetItem(dict, error_str(), msg));
            std::ostringstream stm;
            stm << inst;
            auto str = stm.str();
            py_object py_str(pyunicode_from_string(str.c_str()));
            throw_if(PyDict_SetItem(dict, inst_str(), py_str));
        }

        void GLUT(uint8_t chn_mask, const uint16_t *gaddrs, const uint16_t *starts,
                  const uint16_t *ends, int cnt)
        {
            throw_if(PyDict_SetItem(dict, type_str(), glut_str()));
            set_channels(chn_mask);
            throw_if(PyDict_SetItem(dict, count_str(), pylong_cached(cnt)));
            py_object py_gaddrs(pylist_new(cnt));
            py_object py_starts(pylist_new(cnt));
            py_object py_ends(pylist_new(cnt));
            for (int i = 0; i < cnt; i++) {
                PyList_SET_ITEM(py_gaddrs.get(), i, pylong_from_long(gaddrs[i]));
                PyList_SET_ITEM(py_starts.get(), i, pylong_from_long(starts[i]));
                PyList_SET_ITEM(py_ends.get(), i, pylong_from_long(ends[i]));
            }
            throw_if(PyDict_SetItem(dict, gaddrs_str(), py_gaddrs));
            throw_if(PyDict_SetItem(dict, starts_str(), py_starts));
            throw_if(PyDict_SetItem(dict, ends_str(), py_ends));
        }

        void SLUT(uint8_t chn_mask, const uint16_t *saddrs,
                  const ModTypeMask *mod_types, const uint16_t *paddrs, int cnt)
        {
            throw_if(PyDict_SetItem(dict, type_str(), slut_str()));
            set_channels(chn_mask);
            throw_if(PyDict_SetItem(dict, count_str(), pylong_cached(cnt)));
            py_object py_saddrs(pylist_new(cnt));
            py_object py_modtypes(pylist_new(cnt));
            py_object py_paddrs(pylist_new(cnt));
            for (int i = 0; i < cnt; i++) {
                PyList_SET_ITEM(py_saddrs.get(), i, pylong_from_long(saddrs[i]));
                PyList_SET_ITEM(py_paddrs.get(), i, pylong_from_long(paddrs[i]));
                PyList_SET_ITEM(py_modtypes.get(), i,
                                mod_type_list(mod_types[i]).release());
            }
            throw_if(PyDict_SetItem(dict, saddrs_str(), py_saddrs));
            throw_if(PyDict_SetItem(dict, modtypes_str(), py_modtypes));
            throw_if(PyDict_SetItem(dict, paddrs_str(), py_paddrs));
        }
        void GSEQ(uint8_t chn_mask, const uint16_t *gaddrs, int cnt, SeqMode m)
        {
            if (m == SeqMode::GATE) {
                throw_if(PyDict_SetItem(dict, type_str(), gseq_str()));
            }
            else if (m == SeqMode::WAIT_ANC) {
                throw_if(PyDict_SetItem(dict, type_str(), wait_anc_str()));
            }
            else if (m == SeqMode::CONT_ANC) {
                throw_if(PyDict_SetItem(dict, type_str(), cont_anc_str()));
            }
            set_channels(chn_mask);
            throw_if(PyDict_SetItem(dict, count_str(), pylong_cached(cnt)));
            py_object py_gaddrs(pylist_new(cnt));
            for (int i = 0; i < cnt; i++)
                PyList_SET_ITEM(py_gaddrs.get(), i, pylong_from_long(gaddrs[i]));
            throw_if(PyDict_SetItem(dict, gaddrs_str(), py_gaddrs));
        }

        void pulse(int chn_mask, ModTypeMask mod_type, const PDQSpline &spl,
                   int64_t cycles, Executor::PulseMeta meta, Executor::PulseTarget tgt)
        {
            if (tgt.type == Executor::PulseTarget::None) {
                throw_if(PyDict_SetItem(dict, type_str(), pulse_data_str()));
            }
            else if (tgt.type == Executor::PulseTarget::PLUT) {
                throw_if(PyDict_SetItem(dict, type_str(), plut_str()));
            }
            else if (tgt.type == Executor::PulseTarget::Stream) {
                throw_if(PyDict_SetItem(dict, type_str(), stream_str()));
            }
            bool is_plut = tgt.type == Executor::PulseTarget::PLUT;
            if (is_plut) {
                py_object paddr(pylong_from_long(tgt.addr));
                throw_if(PyDict_SetItem(dict, paddr_str(), paddr));
            }
            else {
                throw_if(PyDict_SetItem(dict, modtype_str(), mod_type_list(mod_type)));
            }
            set_channels(chn_mask);
            py_object py_cycles(pylong_from_longlong(cycles));
            throw_if(PyDict_SetItem(dict, cycles_str(), py_cycles));
            py_object py_spl(pylist_new(4));
            for (int i = 0; i < 4; i++)
                PyList_SET_ITEM(py_spl.get(), i, pylong_from_longlong(spl.orders[i]));
            throw_if(PyDict_SetItem(dict, spline_mu_str(), py_spl));
            throw_if(PyDict_SetItem(dict, spline_shift_str(), pylong_cached(spl.shift)));
            bool freq = mod_type & (FRQMOD0_MASK | FRQMOD1_MASK) || is_plut;
            bool amp = mod_type & (AMPMOD0_MASK | AMPMOD1_MASK) || is_plut;
            bool phase = mod_type & (PHSMOD0_MASK | PHSMOD1_MASK |
                                     FRMROT0_MASK | FRMROT1_MASK) || is_plut;
            bool unprefix_spline = (int(freq) + int(amp) + int(phase)) == 1;
            auto set_spline = [&] (PyObject *name, auto &&cb, bool cond) {
                if (!cond)
                    return;
                auto fspl = cb(spl, cycles);
                py_object py_fspl(pylist_new(4));
                PyList_SET_ITEM(py_fspl.get(), 0, pyfloat_from_double(fspl.order0));
                PyList_SET_ITEM(py_fspl.get(), 1, pyfloat_from_double(fspl.order1));
                PyList_SET_ITEM(py_fspl.get(), 2, pyfloat_from_double(fspl.order2));
                PyList_SET_ITEM(py_fspl.get(), 3, pyfloat_from_double(fspl.order3));
                if (unprefix_spline)
                    throw_if(PyDict_SetItem(dict, spline_str(), py_fspl));
                throw_if(PyDict_SetItem(dict, name, py_fspl));
            };
            set_spline(spline_freq_str(), Executor::freq_spline, freq);
            set_spline(spline_amp_str(), Executor::amp_spline, amp);
            set_spline(spline_phase_str(), Executor::phase_spline, phase);
            throw_if(PyDict_SetItem(dict, trig_str(), meta.trig ? Py_True : Py_False));
            throw_if(PyDict_SetItem(dict, sync_str(), meta.sync ? Py_True : Py_False));
            throw_if(PyDict_SetItem(dict, enable_str(), meta.en ? Py_True : Py_False));
            throw_if(PyDict_SetItem(dict, ff_str(), meta.fb ? Py_True : Py_False));
            throw_if(PyDict_SetItem(dict, eof_str(), meta.apply_eof ? Py_True : Py_False));
            throw_if(PyDict_SetItem(dict, clr_str(), meta.clr_frame ? Py_True : Py_False));
            throw_if(PyDict_SetItem(dict, fwd_str(), pylong_cached(meta.fwd_frame_mask)));
            throw_if(PyDict_SetItem(dict, inv_str(), pylong_cached(meta.inv_frame_mask)));
        }

        py_object dict{pydict_new()};

    private:
        void set_channels(uint8_t chn_mask)
        {
            int nchns = std::popcount(chn_mask);
            int chn_added = 0;
            py_object chns(pylist_new(nchns));
            for (int chn = 0; chn < 8; chn++) {
                if (!((chn_mask >> chn) & 1))
                    continue;
                if (nchns == 1)
                    throw_if(PyDict_SetItem(dict, channel_str(), pylong_cached(chn)));
                PyList_SET_ITEM(chns.get(), chn_added, pylong_from_long(chn));
                chn_added++;
            }
            assert(nchns == chn_added);
            throw_if(PyDict_SetItem(dict, channels_str(), chns));
        }
        py_object mod_type_list(ModTypeMask mod_type)
        {
            int nmasks = std::popcount(uint8_t(mod_type));
            py_object names(pylist_new(nmasks));
            int name_added = 0;
            auto add_name = [&] (auto &&cb, ModTypeMask mask) {
                if (!(mod_type & mask))
                    return;
                PyList_SET_ITEM(names.get(), name_added, py_newref(cb()));
                name_added++;
            };
            add_name(freq0_str, FRQMOD0_MASK);
            add_name(amp0_str, AMPMOD0_MASK);
            add_name(phase0_str, PHSMOD0_MASK);
            add_name(freq1_str, FRQMOD1_MASK);
            add_name(amp1_str, AMPMOD1_MASK);
            add_name(phase1_str, PHSMOD1_MASK);
            add_name(frame_rot0_str, FRMROT0_MASK);
            add_name(frame_rot1_str, FRMROT1_MASK);
            assert(nmasks == name_added);
            return names;
        }
    };

    struct PulseSequencer {
        PulseSequencer(bool single_action)
            : single_action(single_action)
        {
            reset();
        }

        void reset()
        {
            memset(_channel_mem, 0xff, sizeof(channels));
        }

        void invalid(const JaqalInst&, Executor::Error)
        {
            pulses.push_back(JaqalInst::mask(0, 255));
        }

        void next()
        {
        }

        void GLUT(uint8_t chn_mask, const uint16_t *gaddrs, const uint16_t *starts,
                  const uint16_t *ends, int cnt)
        {
            for (int chn = 0; chn < 8; chn++) {
                if (!((chn_mask >> chn) & 1))
                    continue;
                auto &glut = channels[chn].glut;
                for (int i = 0; i < cnt; i++) {
                    glut[gaddrs[i]] = { starts[i], ends[i] };
                }
            }
        }
        void SLUT(uint8_t chn_mask, const uint16_t *saddrs,
                  const ModTypeMask *mod_types, const uint16_t *paddrs, int cnt)
        {
            for (int chn = 0; chn < 8; chn++) {
                if (!((chn_mask >> chn) & 1))
                    continue;
                auto &slut = channels[chn].slut;
                for (int i = 0; i < cnt; i++) {
                    slut[saddrs[i]] = { mod_types[i], paddrs[i] };
                }
            }
        }
        void GSEQ(uint8_t chn_mask, const uint16_t *gaddrs, int cnt, SeqMode m)
        {
            if (m != SeqMode::GATE) {
                pulses.push_back(JaqalInst::mask(0, 255));
                return;
            }
            for (int chn = 0; chn < 8; chn++) {
                if (!((chn_mask >> chn) & 1))
                    continue;
                auto &channel = channels[chn];
                for (auto gaddr: std::span(gaddrs, cnt)) {
                    auto [start, end] = channel.glut[gaddr];
                    if (start >= 8192 || end >= 8192 || end < start) {
                        pulses.push_back(JaqalInst::mask(0, 255));
                        continue;
                    }
                    for (auto i = start; i <= end; i++) {
                        auto [mod_type, paddr] = channel.slut[i];
                        if (paddr >= 4096) {
                            pulses.push_back(JaqalInst::mask(0, 255));
                        }
                        else {
                            auto pulse = (channel.plut[paddr] & modtype_nmask &
                                          channel_nmask);
                            pulse = apply_channel_mask(pulse, uint8_t(1 << chn));
                            if (single_action) {
                                for (int bit = 0; bit < 8; bit++) {
                                    if (!((uint8_t(mod_type) >> bit) & 1))
                                        continue;
                                    pulses.push_back(apply_modtype_mask(
                                                         pulse, ModTypeMask(1 << bit)));
                                }
                            }
                            else {
                                pulses.push_back(apply_modtype_mask(pulse, mod_type));
                            }
                        }
                    }
                }
            }
        }

        void pulse(int chn_mask, ModTypeMask mod_type, const PDQSpline &spl,
                   int64_t cycles, Executor::PulseMeta meta, Executor::PulseTarget tgt)
        {
            // use the frame rotation metadata since it is a superset of
            // the normal parameter one.
            auto metadata =
                raw_frame_metadata(spl.shift, meta.trig, meta.apply_eof, meta.clr_frame,
                                   meta.fwd_frame_mask, meta.inv_frame_mask);
            std::array<int64_t,4> sp(spl.orders);
            for (auto &v: sp)
                v &= (int64_t(1) << 40) - 1;
            auto inst = Jaqal_v1_3::pulse(metadata, sp, cycles);
            for (int chn = 0; chn < 8; chn++) {
                if (!((chn_mask >> chn) & 1))
                    continue;
                if (tgt.type == Executor::PulseTarget::PLUT) {
                    channels[chn].plut[tgt.addr] = inst;
                }
                else {
                    auto chninst = apply_channel_mask(inst, uint8_t(1 << chn));
                    if (single_action) {
                        for (int bit = 0; bit < 8; bit++) {
                            if (!((uint8_t(mod_type) >> bit) & 1))
                                continue;
                            pulses.push_back(apply_modtype_mask(
                                                 chninst, ModTypeMask(1 << bit)));
                        }
                    }
                    else {
                        pulses.push_back(apply_modtype_mask(chninst, mod_type));
                    }
                }
            }
        }

        std::vector<JaqalInst> pulses;
        const bool single_action;
    private:
        struct ChannelLUT {
            std::array<JaqalInst,4096> plut;
            std::array<std::pair<ModTypeMask,uint16_t>,8192> slut;
            std::array<std::pair<uint16_t,uint16_t>,4096> glut;
        };
        union {
            ChannelLUT channels[8];
            // std::memset only requires trivially-copyable
            // but GCC decides to warn as long as the types are non-trivial
            // (preventing us from defining default constructors)
            // Work around this requirement with a union char buffer.
            char _channel_mem[8 * sizeof(ChannelLUT)];
        };
    };

    static void print_inst(std::ostream &io, const JaqalInst &inst, bool print_float)
    {
        Printer printer{io, print_float};
        Executor::execute(printer, inst);
    }

    static void print_insts(std::ostream &io, const char *p, size_t sz, bool print_float)
    {
        Printer printer{io, print_float};
        Executor::execute(printer, std::span(p, sz));
    }

    static auto extract_pulses(const char *p, size_t sz, bool single_action)
    {
        PulseSequencer sequencer(single_action);
        Executor::execute(sequencer, std::span(p, sz));
        return sequencer.pulses;
    }

    __attribute__((returns_nonnull))
    static PyObject *inst_to_dict(const JaqalInst &inst)
    {
        DictConverter converter;
        Executor::execute(converter, inst);
        return converter.dict.release();
    }

    // Set the minimum clock cycles for a pulse to help avoid underflows. This time
    // is determined by state machine transitions for loading another gate, but does
    // not account for serialization of pulse words.
    static constexpr int MINIMUM_PULSE_CLOCK_CYCLES = 4;
};

struct SyncChannelGen: Generator {
    virtual void add_tone_data(int chn, int64_t duration_cycles, cubic_spline_t freq,
                               cubic_spline_t amp, cubic_spline_t phase,
                               output_flags_t flags, int64_t cur_cycle) = 0;
    void process_channel(ToneBuffer &tone_buffer, int chn,
                         int64_t total_cycle) override;
};

struct PulseCompilerGen: SyncChannelGen {
    struct Info {
        PyObject *channel_list[64];
        PyObject *CubicSpline;
        PyObject *ToneData;
        PyObject *cubic_0;
        std::vector<std::pair<PyObject*,PyObject*>> tonedata_fields;
        PyObject *channel_key;
        PyObject *tone_key;
        PyObject *duration_cycles_key;
        PyObject *frequency_hz_key;
        PyObject *amplitude_key;
        PyObject *phase_rad_key;
        PyObject *frame_rotation_rad_key;
        PyObject *wait_trigger_key;
        PyObject *sync_key;
        PyObject *output_enable_key;
        PyObject *feedback_enable_key;
        PyObject *bypass_lookup_tables_key;

        static inline void dict_setitem(PyObject *dict, PyObject *key, PyObject *value)
        {
            throw_if(PyDict_SetItem(dict, key, value));
        }

        __attribute__((returns_nonnull,always_inline))
        PyObject *_new_cubic_spline(cubic_spline_t sp)
        {
            PyTypeObject *ty = (PyTypeObject*)CubicSpline;
            py_object o0(pyfloat_from_double(sp.order0));
            py_object o1(pyfloat_from_double(sp.order1));
            py_object o2(pyfloat_from_double(sp.order2));
            py_object o3(pyfloat_from_double(sp.order3));
            auto newobj = pytype_genericalloc(ty, 4);
            PyTuple_SET_ITEM(newobj, 0, o0.release());
            PyTuple_SET_ITEM(newobj, 1, o1.release());
            PyTuple_SET_ITEM(newobj, 2, o2.release());
            PyTuple_SET_ITEM(newobj, 3, o3.release());
            return newobj;
        }

        inline __attribute__((returns_nonnull))
        PyObject *new_cubic_spline(cubic_spline_t sp)
        {
            if (sp == cubic_spline_t{0, 0, 0, 0})
                return py_newref(cubic_0);
            return _new_cubic_spline(sp);
        }

        py_object new_tone_data(int channel, int tone, int64_t duration_cycles,
                                cubic_spline_t freq, cubic_spline_t amp,
                                cubic_spline_t phase, output_flags_t flags)
        {
            py_object td(pytype_genericalloc(ToneData));
            py_object td_dict(throw_if_not(PyObject_GenericGetDict(td, nullptr)));
            for (auto [key, value]: tonedata_fields)
                dict_setitem(td_dict, key, value);
            static_assert(_pylong_cache_max >= 32);
            dict_setitem(td_dict, channel_key, pylong_cached(channel));
            dict_setitem(td_dict, tone_key, pylong_cached(tone));
            {
                py_object py_cycles(pylong_from_longlong(duration_cycles));
                dict_setitem(td_dict, duration_cycles_key, py_cycles);
            }
            {
                py_object py_freq(new_cubic_spline(freq));
                dict_setitem(td_dict, frequency_hz_key, py_freq);
            }
            {
                py_object py_amp(new_cubic_spline(amp));
                dict_setitem(td_dict, amplitude_key, py_amp);
            }
            {
                // tone data wants rad as phase unit.
                py_object py_phase(new_cubic_spline({
                            phase.order0 * (2 * M_PI), phase.order1 * (2 * M_PI),
                            phase.order2 * (2 * M_PI), phase.order3 * (2 * M_PI) }));
                dict_setitem(td_dict, phase_rad_key, py_phase);
            }
            dict_setitem(td_dict, frame_rotation_rad_key, cubic_0);
            dict_setitem(td_dict, wait_trigger_key, flags.wait_trigger ? Py_True : Py_False);
            dict_setitem(td_dict, sync_key, flags.sync ? Py_True : Py_False);
            dict_setitem(td_dict, output_enable_key, Py_False);
            dict_setitem(td_dict, feedback_enable_key, flags.feedback_enable ? Py_True : Py_False);
            dict_setitem(td_dict, bypass_lookup_tables_key, Py_False);
            return td;
        }

        Info();
    };
    static inline Info *get_info()
    {
        static Info info;
        return &info;
    }

    void add_tone_data(int chn, int64_t duration_cycles, cubic_spline_t freq,
                       cubic_spline_t amp, cubic_spline_t phase,
                       output_flags_t flags, int64_t) override
    {
        bb_debug("outputting tone data: chn=%d, cycles=%" PRId64 ", sync=%d, ff=%d\n",
                 chn, duration_cycles, flags.sync, flags.feedback_enable);
        auto info = get_info();
        auto tonedata = info->new_tone_data(chn >> 1, chn & 1, duration_cycles, freq,
                                            amp, phase, flags);
        auto key = info->channel_list[chn];
        PyObject *tonedatas;
        if (last_chn == chn) [[likely]] {
            tonedatas = assume(last_tonedatas);
        }
        else {
            tonedatas = PyDict_GetItemWithError(output, key);
        }
        if (!tonedatas) {
            throw_if(PyErr_Occurred());
            py_object tonedatas(pylist_new(1));
            PyList_SET_ITEM(tonedatas.get(), 0, tonedata.release());
            throw_if(PyDict_SetItem(output, key, tonedatas));
            last_tonedatas = tonedatas.get();
        }
        else {
            pylist_append(tonedatas, tonedata);
            last_tonedatas = tonedatas;
        }
        last_chn = chn;
    }

    PulseCompilerGen()
        : output(pydict_new())
    {
    }
    void start() override
    {
        PyDict_Clear(output);
        last_chn = -1;
    }
    void end() override
    {}

    py_object output;
    int last_chn;
    PyObject *last_tonedatas;
};

Generator *new_pulse_compiler_generator()
{
    return new PulseCompilerGen;
}

PulseCompilerGen::Info::Info()
    : channel_key(pyunicode_from_string("channel")),
      tone_key(pyunicode_from_string("tone")),
      duration_cycles_key(pyunicode_from_string("duration_cycles")),
      frequency_hz_key(pyunicode_from_string("frequency_hz")),
      amplitude_key(pyunicode_from_string("amplitude")),
      phase_rad_key(pyunicode_from_string("phase_rad")),
      frame_rotation_rad_key(pyunicode_from_string("frame_rotation_rad")),
      wait_trigger_key(pyunicode_from_string("wait_trigger")),
      sync_key(pyunicode_from_string("sync")),
      output_enable_key(pyunicode_from_string("output_enable")),
      feedback_enable_key(pyunicode_from_string("feedback_enable")),
      bypass_lookup_tables_key(pyunicode_from_string("bypass_lookup_tables"))
{
    py_object tonedata_mod(
        throw_if_not(PyImport_ImportModule("pulsecompiler.rfsoc.tones.tonedata")));
    ToneData = throw_if_not(PyObject_GetAttrString(tonedata_mod, "ToneData"));
    py_object splines_mod(
        throw_if_not(PyImport_ImportModule("pulsecompiler.rfsoc.structures.splines")));
    CubicSpline = throw_if_not(PyObject_GetAttrString(splines_mod, "CubicSpline"));
    cubic_0 = _new_cubic_spline({0, 0, 0, 0});
    py_object pulse_mod(throw_if_not(PyImport_ImportModule("qiskit.pulse")));
    py_object ControlChannel(throw_if_not(PyObject_GetAttrString(pulse_mod,
                                                                 "ControlChannel")));
    py_object DriveChannel(throw_if_not(PyObject_GetAttrString(pulse_mod,
                                                               "DriveChannel")));

    static_assert(_pylong_cache_max >= 64);
    PyObject *py_nums[64];
    for (int i = 0; i < 64; i++)
        py_nums[i] = pylong_cached(i);

    channel_list[0] = throw_if_not(
        _PyObject_Vectorcall(ControlChannel, &py_nums[0], 1, nullptr));
    channel_list[1] = throw_if_not(
        _PyObject_Vectorcall(ControlChannel, &py_nums[1], 1, nullptr));
    for (int i = 0; i < 62; i++)
        channel_list[i + 2] = throw_if_not(
            _PyObject_Vectorcall(DriveChannel, &py_nums[i], 1, nullptr));

    py_object orig_post_init(
        throw_if_not(PyObject_GetAttrString(ToneData, "__post_init__")));
    auto dummy_post_init_cb = [] (PyObject*, PyObject *const*, Py_ssize_t) {
        return py_immref(Py_None);
    };
    static PyMethodDef dummy_post_init_method = {
        "__post_init__", (PyCFunction)(void*)(_PyCFunctionFast)dummy_post_init_cb,
        METH_FASTCALL, 0};
    py_object dummy_post_init(PyCFunction_New(&dummy_post_init_method, nullptr));
    throw_if(PyObject_SetAttrString(ToneData, "__post_init__", dummy_post_init));
    py_object dummy_tonedata(
        throw_if_not(_PyObject_Vectorcall(ToneData, py_nums, 6, nullptr)));
    throw_if(PyObject_SetAttrString(ToneData, "__post_init__", orig_post_init));
    py_object td_dict(throw_if_not(PyObject_GenericGetDict(dummy_tonedata, nullptr)));
    for (auto [key, value]: pydict_iter(td_dict)) {
        for (auto name: {"channel", "tone", "duration_cycles", "frequency_hz",
                "amplitude", "phase_rad", "frame_rotation_rad", "wait_trigger",
                "sync", "output_enable", "feedback_enable",
                "bypass_lookup_tables"}) {
            if (PyUnicode_CompareWithASCIIString(key, name) == 0) {
                goto skip_key;
            }
        }
        tonedata_fields.push_back({ { py_newref(key) } , py_newref(value) });
    skip_key:
        ;
    }
}

struct JaqalPulseCompilerGen: SyncChannelGen {
    struct BoardGen {
        Jaqal_v1::ChannelGen channels[8];
        void clear()
        {
            for (auto &channel: channels) {
                channel.clear();
            }
        }
        PyObject *get_prefix() const;
        PyObject *get_sequence() const;
        void end();
    };
    BoardGen boards[4]; // 4 * 8 physical channels

    void start() override
    {
        for (auto &board: boards) {
            board.clear();
        }
    }
    void add_tone_data(int chn, int64_t duration_cycles, cubic_spline_t freq,
                       cubic_spline_t amp, cubic_spline_t phase,
                       output_flags_t flags, int64_t cur_cycle) override;
    void end() override
    {
#pragma omp parallel
#pragma omp single
#pragma omp taskloop
        for (auto &board: boards) {
            board.end();
        }
    }

    __attribute__((returns_nonnull)) PyObject *get_prefix(int n) const
    {
        if (n < 0 || n >= 4)
            throw std::out_of_range("Board index should be in [0, 3]");
        return boards[n].get_prefix();
    }

    __attribute__((returns_nonnull)) PyObject *get_sequence(int n) const
    {
        if (n < 0 || n >= 4)
            throw std::out_of_range("Board index should be in [0, 3]");
        return boards[n].get_sequence();
    }
};

Generator *new_jaqal_pulse_compiler_generator()
{
    return new JaqalPulseCompilerGen;
}

__attribute__((flatten))
static inline void chn_add_tone_data(auto &channel_gen, int channel, int tone,
                                     int64_t duration_cycles,
                                     cubic_spline_t freq, cubic_spline_t amp,
                                     cubic_spline_t phase, output_flags_t flags,
                                     int64_t cur_cycle)
{
    assume(tone == 0 || tone == 1);
    channel_gen.add_pulse(Jaqal_v1::freq_pulse(channel, tone, freq, duration_cycles,
                                               flags.wait_trigger, flags.sync,
                                               flags.feedback_enable), cur_cycle);
    channel_gen.add_pulse(Jaqal_v1::amp_pulse(channel, tone, amp, duration_cycles,
                                              flags.wait_trigger), cur_cycle);
    channel_gen.add_pulse(Jaqal_v1::phase_pulse(channel, tone, phase, duration_cycles,
                                                flags.wait_trigger), cur_cycle);
    channel_gen.add_pulse(Jaqal_v1::frame_pulse(channel, tone, {0, 0, 0, 0},
                                                duration_cycles, flags.wait_trigger,
                                                false, false, 0, 0), cur_cycle);
}

void JaqalPulseCompilerGen::add_tone_data(int chn, int64_t duration_cycles,
                                          cubic_spline_t freq, cubic_spline_t amp,
                                          cubic_spline_t phase, output_flags_t flags,
                                          int64_t cur_cycle)
{
    auto board_id = chn >> 4;
    assert(board_id < 4);
    auto &board_gen = boards[board_id];
    auto channel = (chn >> 1) & 7;
    auto tone = chn & 1;
    auto &channel_gen = board_gen.channels[channel];
    int64_t max_cycles = (int64_t(1) << 40) - 1;
    auto clear_edge_flags = [] (auto &flags) {
        flags.wait_trigger = false;
        flags.sync = false;
    };
    if (duration_cycles > max_cycles) [[unlikely]] {
        int64_t tstart = 0;
        auto resample = [&] (auto spline, int64_t tstart, int64_t tend) {
            return spline_resample_cycle(spline, 0, duration_cycles, tstart, tend);
        };
        while ((duration_cycles - tstart) > max_cycles * 2) {
            int64_t tend = tstart + max_cycles;
            chn_add_tone_data(channel_gen, channel, tone, max_cycles,
                              resample(freq, tstart, tend),
                              resample(amp, tstart, tend),
                              resample(phase, tstart, tend),
                              flags, cur_cycle + tstart);
            clear_edge_flags(flags);
            tstart = tend;
        }
        int64_t tmid = (duration_cycles - tstart) / 2 + tstart;
        chn_add_tone_data(channel_gen, channel, tone, tmid - tstart,
                          resample(freq, tstart, tmid),
                          resample(amp, tstart, tmid),
                          resample(phase, tstart, tmid),
                          flags, cur_cycle + tstart);
        clear_edge_flags(flags);
        chn_add_tone_data(channel_gen, channel, tone, duration_cycles - tmid,
                          resample(freq, tmid, duration_cycles),
                          resample(amp, tmid, duration_cycles),
                          resample(phase, tmid, duration_cycles),
                          flags, cur_cycle + tmid);
        return;
    }
    chn_add_tone_data(channel_gen, channel, tone, duration_cycles,
                      freq, amp, phase, flags, cur_cycle);
}

PyObject *JaqalPulseCompilerGen::BoardGen::get_prefix() const
{
    pybytes_ostream io;
    for (int chn = 0; chn < 8; chn++) {
        auto &channel_gen = channels[chn];
        for (auto &[pulse, addr]: channel_gen.pulses.pulses) {
            auto inst = Jaqal_v1::program_PLUT(pulse, addr);
            io.write((char*)&inst, sizeof(inst));
        }
        uint16_t idxbuff[std::max(Jaqal_v1::SLUT_MAXCNT, Jaqal_v1::GLUT_MAXCNT)];
        for (int i = 0; i < sizeof(idxbuff) / sizeof(uint16_t); i++)
            idxbuff[i] = i;
        auto nslut = (int)channel_gen.slut.size();
        for (int i = 0; i < nslut; i += Jaqal_v1::SLUT_MAXCNT) {
            auto blksize = std::min(Jaqal_v1::SLUT_MAXCNT, nslut - i);
            for (int j = 0; j < blksize; j++)
                idxbuff[j] = i + j;
            auto inst = Jaqal_v1::program_SLUT(chn, idxbuff,
                                               (const uint16_t*)&channel_gen.slut[i],
                                               blksize);
            io.write((char*)&inst, sizeof(inst));
        }
        auto nglut = (int)channel_gen.glut.size();
        uint16_t starts[Jaqal_v1::GLUT_MAXCNT];
        uint16_t ends[Jaqal_v1::GLUT_MAXCNT];
        for (int i = 0; i < nglut; i += Jaqal_v1::GLUT_MAXCNT) {
            auto blksize = std::min(Jaqal_v1::GLUT_MAXCNT, nglut - i);
            for (int j = 0; j < blksize; j++) {
                auto [start, end] = channel_gen.glut[i + j];
                starts[j] = start;
                ends[j] = end;
                idxbuff[j] = i + j;
            }
            auto inst = Jaqal_v1::program_GLUT(chn, idxbuff, starts, ends, blksize);
            io.write((char*)&inst, sizeof(inst));
        }
    }
    return io.get_buf();
}

PyObject *JaqalPulseCompilerGen::BoardGen::get_sequence() const
{
    pybytes_ostream io;
    std::span<const TimedID> chn_gate_ids[8];
    for (int chn = 0; chn < 8; chn++)
        chn_gate_ids[chn] = std::span(channels[chn].gate_ids);
    auto output_channel = [&] (int chn) {
        uint16_t gaddrs[Jaqal_v1::GSEQ_MAXCNT];
        auto &gate_ids = chn_gate_ids[chn];
        assert(gate_ids.size() != 0);
        int blksize = std::min(Jaqal_v1::GSEQ_MAXCNT, (int)gate_ids.size());
        for (int i = 0; i < blksize; i++)
            gaddrs[i] = gate_ids[i].id;
        auto inst = Jaqal_v1::sequence(chn, Jaqal_v1::SeqMode::GATE, gaddrs, blksize);
        io.write((char*)&inst, sizeof(inst));
        gate_ids = gate_ids.subspan(blksize);
    };
    while (true) {
        int out_chn = -1;
        int64_t out_time = INT64_MAX;
        for (int chn = 0; chn < 8; chn++) {
            auto &gate_ids = chn_gate_ids[chn];
            if (gate_ids.size() == 0)
                continue;
            auto first_time = gate_ids[0].time;
            if (first_time < out_time) {
                out_chn = chn;
                out_time = first_time;
            }
        }
        if (out_chn < 0)
            break;
        output_channel(out_chn);
    }
    return io.get_buf();
}

void JaqalPulseCompilerGen::BoardGen::end()
{
#pragma omp taskloop
    for (auto &channel_gen: channels) {
        channel_gen.end();
    }
}

void SyncChannelGen::process_channel(ToneBuffer &tone_buffer, int chn,
                                     int64_t total_cycle)
{
    IsFirst trig;
    assert(!tone_buffer.params[0].empty());
    assert(!tone_buffer.params[1].empty());
    assert(!tone_buffer.params[2].empty());

    bb_debug("Start outputting tone data for channel %d\n", chn);

    int64_t cur_cycle = 0;

    int64_t freq_cycle = 0;
    int freq_idx = 0;
    auto freq_action = tone_buffer.params[(int)ToneFreq][freq_idx];
    int64_t freq_end_cycle = freq_cycle + freq_action.cycle_len;

    int64_t phase_cycle = 0;
    int phase_idx = 0;
    auto phase_action = tone_buffer.params[(int)TonePhase][phase_idx];
    int64_t phase_end_cycle = phase_cycle + phase_action.cycle_len;

    int64_t amp_cycle = 0;
    int amp_idx = 0;
    auto amp_action = tone_buffer.params[(int)ToneAmp][amp_idx];
    int64_t amp_end_cycle = amp_cycle + amp_action.cycle_len;

    int64_t ff_cycle = 0;
    int ff_idx = 0;
    auto ff_action = tone_buffer.ff[ff_idx];
    int64_t ff_end_cycle = ff_cycle + ff_action.cycle_len;

    while (true) {
        // First figure out if we are starting a new action
        // and how long the current/new action last.
        bool sync = freq_cycle == cur_cycle && freq_action.sync;
        int64_t action_end_cycle = std::min({ freq_end_cycle, phase_end_cycle,
                amp_end_cycle, ff_end_cycle });
        bb_debug("find continuous range [%" PRId64 ", %" PRId64 "] on channel %d\n",
                 cur_cycle, action_end_cycle, chn);

        auto forward_freq = [&] {
            assert(freq_idx + 1 < tone_buffer.params[(int)ToneFreq].size());
            freq_cycle = freq_end_cycle;
            freq_idx += 1;
            freq_action = tone_buffer.params[(int)ToneFreq][freq_idx];
            freq_end_cycle = freq_cycle + freq_action.cycle_len;
        };
        auto forward_amp = [&] {
            assert(amp_idx + 1 < tone_buffer.params[(int)ToneAmp].size());
            amp_cycle = amp_end_cycle;
            amp_idx += 1;
            amp_action = tone_buffer.params[(int)ToneAmp][amp_idx];
            amp_end_cycle = amp_cycle + amp_action.cycle_len;
        };
        auto forward_phase = [&] {
            assert(phase_idx + 1 < tone_buffer.params[(int)TonePhase].size());
            phase_cycle = phase_end_cycle;
            phase_idx += 1;
            phase_action = tone_buffer.params[(int)TonePhase][phase_idx];
            phase_end_cycle = phase_cycle + phase_action.cycle_len;
        };
        auto forward_ff = [&] {
            assert(ff_idx + 1 < tone_buffer.ff.size());
            ff_cycle = ff_end_cycle;
            ff_idx += 1;
            ff_action = tone_buffer.ff[ff_idx];
            ff_end_cycle = ff_cycle + ff_action.cycle_len;
        };

        if (action_end_cycle >= cur_cycle + 4) {
            // There's enough space to output a full tone data.
            auto resample_action_spline = [&] (auto action, int64_t action_cycle) {
                auto t1 = double(cur_cycle - action_cycle) / action.cycle_len;
                auto t2 = double(action_end_cycle - action_cycle) / action.cycle_len;
                return spline_resample(action.spline, t1, t2);
            };

            bb_debug("continuous range long enough for normal output (channel %d)\n",
                     chn);
            add_tone_data(chn, action_end_cycle - cur_cycle,
                          resample_action_spline(freq_action, freq_cycle),
                          resample_action_spline(amp_action, amp_cycle),
                          resample_action_spline(phase_action, phase_cycle),
                          { trig.get(), sync, ff_action.ff }, cur_cycle);
            cur_cycle = action_end_cycle;
        }
        else {
            // The last action is at least 8 cycles long and we eat up at most
            // 4 cycles from it to handle pending sync so we should have at least
            // 4 cycles if we are hitting the end.
            assert(action_end_cycle != total_cycle);
            assert(cur_cycle + 4 <= total_cycle);
            bb_debug("continuous range too short (channel %d)\n", chn);

            auto eval_param = [&] (auto &param, int64_t cycle, int64_t cycle_start) {
                auto dt = cycle - cycle_start;
                auto len = param.cycle_len;
                if (len == 0) {
                    assert(dt == 0);
                    return param.spline.order0;
                }
                return spline_eval(param.spline, double(dt) / len);
            };

            // Now we don't have enough time to do a tone data
            // based on the segmentation given to us. We'll manually iterate over
            // the next 4 cycles and compute a 4 cycle tone data that approximate
            // the action we need the closest.

            // This is the frequency we should sync at.
            // We need to record this exactly.
            // It's even more important than the frequency we left the channel at
            // since getting this wrong could mean a huge phase shift.
            double sync_freq = freq_action.spline.order0;
            double freqs[5];
            while (true) {
                auto min_cycle = (int)std::max(freq_cycle - cur_cycle, int64_t(0));
                auto max_cycle = (int)std::min(freq_end_cycle - cur_cycle, int64_t(4));
                for (int cycle = min_cycle; cycle <= max_cycle; cycle++)
                    freqs[cycle] = eval_param(freq_action, cur_cycle + cycle,
                                              freq_cycle);
                if (freq_end_cycle >= cur_cycle + 4)
                    break;
                forward_freq();
                if (freq_action.sync) {
                    sync = true;
                    sync_freq = freq_action.spline.order0;
                }
            }
            action_end_cycle = freq_end_cycle;
            double phases[5];
            while (true) {
                auto min_cycle = (int)std::max(phase_cycle - cur_cycle, int64_t(0));
                auto max_cycle = (int)std::min(phase_end_cycle - cur_cycle, int64_t(4));
                for (int cycle = min_cycle; cycle <= max_cycle; cycle++)
                    phases[cycle] = eval_param(phase_action, cur_cycle + cycle,
                                               phase_cycle);
                if (phase_end_cycle >= cur_cycle + 4)
                    break;
                forward_phase();
            }
            action_end_cycle = std::min(action_end_cycle, phase_end_cycle);
            double amps[5];
            while (true) {
                auto min_cycle = (int)std::max(amp_cycle - cur_cycle, int64_t(0));
                auto max_cycle = (int)std::min(amp_end_cycle - cur_cycle, int64_t(4));
                for (int cycle = min_cycle; cycle <= max_cycle; cycle++)
                    amps[cycle] = eval_param(amp_action, cur_cycle + cycle,
                                             amp_cycle);
                if (amp_end_cycle >= cur_cycle + 4)
                    break;
                forward_amp();
            }
            action_end_cycle = std::min(action_end_cycle, amp_end_cycle);
            while (true) {
                if (ff_end_cycle >= cur_cycle + 4)
                    break;
                forward_ff();
            }
            action_end_cycle = std::min(action_end_cycle, ff_end_cycle);

            bb_debug("freq: {%f, %f, %f, %f, %f}\n",
                     freqs[0], freqs[1], freqs[2], freqs[3], freqs[4]);
            bb_debug("amp: {%f, %f, %f, %f, %f}\n",
                     amps[0], amps[1], amps[2], amps[3], amps[4]);
            bb_debug("phase: {%f, %f, %f, %f, %f}\n",
                     phases[0], phases[1], phases[2], phases[3], phases[4]);
            bb_debug("cur_cycle=%" PRId64 ", end_cycle=%" PRId64 "\n",
                     cur_cycle, action_end_cycle);

            // We can only sync at the start of the tone data so the start frequency
            // must be the sync frequency.
            if (sync) {
                freqs[0] = sync_freq;
                bb_debug("sync at %f\n", freqs[0]);
            }
            else {
                bb_debug("no sync\n");
            }
            add_tone_data(chn, 4, approximate_spline(freqs),
                          approximate_spline(amps), approximate_spline(phases),
                          { trig.get(), sync, ff_action.ff }, cur_cycle);
            cur_cycle += 4;
            if (cur_cycle != action_end_cycle) {
                // We've only outputted 4 cycles (instead of outputting
                // to the end of an action) so in general there may not be anything
                // to post-process. However, if we happen to be hitting the end
                // of an action on the 4 cycle mark, we need to do the post-processing
                // to maintain the invariance that we are not at the end
                // of the sequence.
                assert(cur_cycle < action_end_cycle);
                continue;
            }
        }
        if (action_end_cycle == total_cycle)
            break;
        if (action_end_cycle == freq_end_cycle)
            forward_freq();
        if (action_end_cycle == amp_end_cycle)
            forward_amp();
        if (action_end_cycle == phase_end_cycle)
            forward_phase();
        if (action_end_cycle == ff_end_cycle)
            forward_ff();
    }
}

struct Jaqalv1_3Generator: Generator {
    virtual void add_inst(const JaqalInst &inst, int board, int board_chn,
                          Jaqal_v1_3::ModType mod, int64_t cycle) = 0;
private:
    struct ChnInfo {
        uint16_t board;
        uint8_t board_chn;
        uint8_t tone;
        ChnInfo(int chn)
            : board(chn / 16),
              board_chn((chn / 2) % 8),
              tone(chn % 2)
        {
        }
    };
    static inline int64_t limit_cycles(int64_t cur, int64_t end)
    {
        int64_t max_cycles = (int64_t(1) << 40) - 1;
        int64_t len = end - cur;
        if (len <= max_cycles)
            return end;
        if (len > max_cycles * 2)
            return cur + max_cycles;
        return cur + len / 2;
    }
    void process_freq(std::span<DDSParamAction> freq, std::span<DDSFFAction> ff,
                      ChnInfo chn, int64_t total_cycle);
    template<typename P>
    void process_param(std::span<DDSParamAction> param, ChnInfo chn,
                       int64_t total_cycle, Jaqal_v1_3::ModType modtype, P &&pulsef);
    void process_frame(ChnInfo chn, int64_t total_cycle, Jaqal_v1_3::ModType modtype);
    void process_channel(ToneBuffer &tone_buffer, int chn, int64_t total_cycle) override;
};

inline __attribute__((always_inline))
void Jaqalv1_3Generator::process_freq(std::span<DDSParamAction> freq_actions,
                                      std::span<DDSFFAction> ff_actions,
                                      ChnInfo chn, int64_t total_cycle)
{
    IsFirst trig;
    assume(chn.tone == 0 || chn.tone == 1);
    Jaqal_v1_3::ModType modtype =
        chn.tone == 0 ? Jaqal_v1_3::FRQMOD0 : Jaqal_v1_3::FRQMOD1;

    int64_t cur_cycle = 0;

    int64_t freq_cycle = 0;
    int freq_idx = 0;
    auto freq_action = freq_actions[freq_idx];
    int64_t freq_end_cycle = freq_cycle + freq_action.cycle_len;

    int64_t ff_cycle = 0;
    int ff_idx = 0;
    auto ff_action = ff_actions[ff_idx];
    int64_t ff_end_cycle = ff_cycle + ff_action.cycle_len;

    while (true) {
        // First figure out if we are starting a new action
        // and how long the current/new action last.
        bool sync = freq_cycle == cur_cycle && freq_action.sync;
        int64_t action_end_cycle =
            limit_cycles(cur_cycle, std::min({ freq_end_cycle, ff_end_cycle }));
        bb_debug("find continuous range [%" PRId64 ", %" PRId64
                 "] for freq on board %d, chn %d, tone %d\n",
                 cur_cycle, action_end_cycle, chn.board, chn.board_chn, chn.tone);

        auto forward_freq = [&] {
            assert(freq_idx + 1 < freq_actions.size());
            freq_cycle = freq_end_cycle;
            freq_idx += 1;
            freq_action = freq_actions[freq_idx];
            freq_end_cycle = freq_cycle + freq_action.cycle_len;
        };
        auto forward_ff = [&] {
            assert(ff_idx + 1 < ff_actions.size());
            ff_cycle = ff_end_cycle;
            ff_idx += 1;
            ff_action = ff_actions[ff_idx];
            ff_end_cycle = ff_cycle + ff_action.cycle_len;
        };

        if (action_end_cycle >= cur_cycle + 4) {
            // There's enough space to output a full tone data.
            auto resample_action_spline = [&] (auto action, int64_t action_cycle) {
                auto t1 = double(cur_cycle - action_cycle) / action.cycle_len;
                auto t2 = double(action_end_cycle - action_cycle) / action.cycle_len;
                return spline_resample(action.spline, t1, t2);
            };

            bb_debug("continuous range long enough for normal freq output\n");
            add_inst(Jaqal_v1_3::freq_pulse(resample_action_spline(freq_action,
                                                                   freq_cycle),
                                            action_end_cycle - cur_cycle, trig.get(),
                                            sync, ff_action.ff),
                     chn.board, chn.board_chn, modtype, cur_cycle);
            cur_cycle = action_end_cycle;
        }
        else {
            // The last action is at least 8 cycles long and we eat up at most
            // 4 cycles from it to handle pending sync so we should have at least
            // 4 cycles if we are hitting the end.
            assert(action_end_cycle != total_cycle);
            assert(cur_cycle + 4 <= total_cycle);
            bb_debug("continuous range too short for freq\n");

            auto eval_param = [&] (auto &param, int64_t cycle, int64_t cycle_start) {
                auto dt = cycle - cycle_start;
                auto len = param.cycle_len;
                if (len == 0) {
                    assert(dt == 0);
                    return param.spline.order0;
                }
                return spline_eval(param.spline, double(dt) / len);
            };

            // Now we don't have enough time to do a tone data
            // based on the segmentation given to us. We'll manually iterate over
            // the next 4 cycles and compute a 4 cycle tone data that approximate
            // the action we need the closest.

            // This is the frequency we should sync at.
            // We need to record this exactly.
            // It's even more important than the frequency we left the channel at
            // since getting this wrong could mean a huge phase shift.
            double sync_freq = freq_action.spline.order0;
            double freqs[5];
            while (true) {
                auto min_cycle = (int)std::max(freq_cycle - cur_cycle, int64_t(0));
                auto max_cycle = (int)std::min(freq_end_cycle - cur_cycle, int64_t(4));
                for (int cycle = min_cycle; cycle <= max_cycle; cycle++)
                    freqs[cycle] = eval_param(freq_action, cur_cycle + cycle,
                                              freq_cycle);
                if (freq_end_cycle >= cur_cycle + 4)
                    break;
                forward_freq();
                if (freq_action.sync) {
                    sync = true;
                    sync_freq = freq_action.spline.order0;
                }
            }
            action_end_cycle = freq_end_cycle;
            while (true) {
                if (ff_end_cycle >= cur_cycle + 4)
                    break;
                forward_ff();
            }
            action_end_cycle = std::min(action_end_cycle, ff_end_cycle);

            bb_debug("freq: {%f, %f, %f, %f, %f}\n",
                     freqs[0], freqs[1], freqs[2], freqs[3], freqs[4]);
            bb_debug("cur_cycle=%" PRId64 ", end_cycle=%" PRId64 "\n",
                     cur_cycle, action_end_cycle);

            // We can only sync at the start of the tone data so the start frequency
            // must be the sync frequency.
            if (sync) {
                freqs[0] = sync_freq;
                bb_debug("sync at %f\n", freqs[0]);
            }
            else {
                bb_debug("no sync\n");
            }
            add_inst(Jaqal_v1_3::freq_pulse(approximate_spline(freqs),
                                            4, trig.get(), sync, ff_action.ff),
                     chn.board, chn.board_chn, modtype, cur_cycle);
            cur_cycle += 4;
            if (cur_cycle != action_end_cycle) {
                // We've only outputted 4 cycles (instead of outputting
                // to the end of an action) so in general there may not be anything
                // to post-process. However, if we happen to be hitting the end
                // of an action on the 4 cycle mark, we need to do the post-processing
                // to maintain the invariance that we are not at the end
                // of the sequence.
                assert(cur_cycle < action_end_cycle);
                continue;
            }
        }
        if (action_end_cycle == total_cycle)
            break;
        if (action_end_cycle == freq_end_cycle)
            forward_freq();
        if (action_end_cycle == ff_end_cycle)
            forward_ff();
    }
}

template<typename P>
inline __attribute__((always_inline))
void Jaqalv1_3Generator::process_param(std::span<DDSParamAction> actions, ChnInfo chn,
                                       int64_t total_cycle, Jaqal_v1_3::ModType modtype,
                                       P &&pulsef)
{
    IsFirst trig;
    int64_t cur_cycle = 0;

    int64_t action_cycle = 0;
    int action_idx = 0;
    auto action = actions[action_idx];
    int64_t action_end_cycle = action_cycle + action.cycle_len;

    while (true) {
        // First figure out if we are starting a new action
        // and how long the current/new action last.
        int64_t block_end_cycle = limit_cycles(cur_cycle, action_end_cycle);

        bb_debug("find continuous range [%" PRId64 ", %" PRId64
                 "] for %d on board %d, chn %d, tone %d\n",
                 cur_cycle, block_end_cycle, int(modtype),
                 chn.board, chn.board_chn, chn.tone);

        auto forward = [&] {
            assert(action_idx + 1 < actions.size());
            action_cycle = action_end_cycle;
            action_idx += 1;
            action = actions[action_idx];
            action_end_cycle = action_cycle + action.cycle_len;
        };

        if (block_end_cycle >= cur_cycle + 4) {
            // There's enough space to output a full tone data.
            auto resample_action_spline = [&] (auto action, int64_t action_cycle) {
                auto t1 = double(cur_cycle - action_cycle) / action.cycle_len;
                auto t2 = double(block_end_cycle - action_cycle) / action.cycle_len;
                return spline_resample(action.spline, t1, t2);
            };

            bb_debug("continuous range long enough for normal output\n");
            add_inst(pulsef(resample_action_spline(action, action_cycle),
                            block_end_cycle - cur_cycle, trig.get(), false, false),
                     chn.board, chn.board_chn, modtype, cur_cycle);
            cur_cycle = block_end_cycle;
        }
        else {
            // The last action is at least 8 cycles long and we eat up at most
            // 4 cycles from it to handle pending sync so we should have at least
            // 4 cycles if we are hitting the end.
            assert(block_end_cycle != total_cycle);
            assert(cur_cycle + 4 <= total_cycle);
            bb_debug("continuous range too short\n");

            auto eval_param = [&] (auto &param, int64_t cycle, int64_t cycle_start) {
                auto dt = cycle - cycle_start;
                auto len = param.cycle_len;
                if (len == 0) {
                    assert(dt == 0);
                    return param.spline.order0;
                }
                return spline_eval(param.spline, double(dt) / len);
            };

            // Now we don't have enough time to do a tone data
            // based on the segmentation given to us. We'll manually iterate over
            // the next 4 cycles and compute a 4 cycle tone data that approximate
            // the action we need the closest.

            double params[5];
            while (true) {
                auto min_cycle = (int)std::max(action_cycle - cur_cycle, int64_t(0));
                auto max_cycle = (int)std::min(action_end_cycle - cur_cycle, int64_t(4));
                for (int cycle = min_cycle; cycle <= max_cycle; cycle++)
                    params[cycle] = eval_param(action, cur_cycle + cycle, action_cycle);
                if (action_end_cycle >= cur_cycle + 4)
                    break;
                forward();
            }
            block_end_cycle = action_end_cycle;

            bb_debug("param: {%f, %f, %f, %f, %f}\n",
                     params[0], params[1], params[2], params[3], params[4]);
            bb_debug("cur_cycle=%" PRId64 ", end_cycle=%" PRId64 "\n",
                     cur_cycle, block_end_cycle);

            add_inst(pulsef(approximate_spline(params), 4, trig.get(), false, false),
                     chn.board, chn.board_chn, modtype, cur_cycle);
            cur_cycle += 4;
            if (cur_cycle != block_end_cycle) {
                // We've only outputted 4 cycles (instead of outputting
                // to the end of an action) so in general there may not be anything
                // to post-process. However, if we happen to be hitting the end
                // of an action on the 4 cycle mark, we need to do the post-processing
                // to maintain the invariance that we are not at the end
                // of the sequence.
                assert(cur_cycle < block_end_cycle);
                continue;
            }
        }
        if (block_end_cycle == total_cycle)
            break;
        if (block_end_cycle == action_end_cycle)
            forward();
    }
}

inline __attribute__((always_inline))
void Jaqalv1_3Generator::process_frame(ChnInfo chn, int64_t total_cycle,
                                       Jaqal_v1_3::ModType modtype)
{
    IsFirst trig;
    int64_t cur_cycle = 0;

    while (cur_cycle < total_cycle) {
        int64_t block_end_cycle = limit_cycles(cur_cycle, total_cycle);

        bb_debug("Add frame rotation for range [%" PRId64 ", %" PRId64
                 "] for %d on board %d, chn %d, tone %d\n",
                 cur_cycle, block_end_cycle, int(modtype),
                 chn.board, chn.board_chn, chn.tone);

        assert(block_end_cycle >= cur_cycle + 4);

        add_inst(Jaqal_v1_3::frame_pulse({0, 0, 0, 0}, block_end_cycle - cur_cycle,
                                         trig.get(), false, false, 0, 0),
                 chn.board, chn.board_chn, modtype, cur_cycle);
        cur_cycle = block_end_cycle;
    }
}

void Jaqalv1_3Generator::process_channel(ToneBuffer &tone_buffer, int chn,
                                         int64_t total_cycle)
{
    bb_debug("Start outputting jaqal v1.3 insts for channel %d\n", chn);
    assert(!tone_buffer.params[0].empty());
    assert(!tone_buffer.params[1].empty());
    assert(!tone_buffer.params[2].empty());

    ChnInfo chninfo{chn};
    process_freq(tone_buffer.params[(int)ToneFreq], tone_buffer.ff,
                 chninfo, total_cycle);
    process_param(tone_buffer.params[(int)ToneAmp], chninfo, total_cycle,
                  chninfo.tone == 0 ? Jaqal_v1_3::AMPMOD0 : Jaqal_v1_3::AMPMOD1,
                  Jaqal_v1_3::amp_pulse);
    process_param(tone_buffer.params[(int)TonePhase], chninfo, total_cycle,
                  chninfo.tone == 0 ? Jaqal_v1_3::PHSMOD0 : Jaqal_v1_3::PHSMOD1,
                  Jaqal_v1_3::phase_pulse);
    process_frame(chninfo, total_cycle,
                  chninfo.tone == 0 ? Jaqal_v1_3::FRMROT0 : Jaqal_v1_3::FRMROT1);
}

struct Jaqalv1_3StreamGen: Jaqalv1_3Generator {
    __attribute__((returns_nonnull)) PyObject *get_prefix(int n) const
    {
        if (n < 0 || n >= 4)
            throw std::out_of_range("Board index should be in [0, 3]");
        static auto empty_bytes = throw_if_not(PyBytes_FromStringAndSize(nullptr, 0));
        return py_newref(empty_bytes);
    }

    __attribute__((returns_nonnull)) PyObject *get_sequence(int n) const
    {
        if (n < 0 || n >= 4)
            throw std::out_of_range("Board index should be in [0, 3]");
        auto &insts = board_insts[n];
        auto ninsts = insts.size();
        static constexpr auto instsz = sizeof(JaqalInst);
        py_object res(throw_if_not(PyBytes_FromStringAndSize(nullptr, ninsts * instsz)));
        auto ptr = PyBytes_AS_STRING(res.get());
        for (size_t i = 0; i < ninsts; i++)
            memcpy(&ptr[i * instsz], &insts[i].inst, instsz);
        return res.release();
    }
private:
    std::vector<TimedInst> board_insts[4];
    void add_inst(const JaqalInst &inst, int board, int board_chn,
                  Jaqal_v1_3::ModType mod, int64_t cycle) override
    {
        assert(board >= 0 && board < 4);
        auto &insts = board_insts[board];
        auto real_inst = Jaqal_v1_3::apply_channel_mask(inst, 1 << board_chn);
        real_inst = Jaqal_v1_3::apply_modtype_mask(real_inst,
                                                   Jaqal_v1_3::ModTypeMask(1 << mod));
        insts.push_back({ cycle, Jaqal_v1_3::stream(real_inst) });
    }
    void start() override
    {
        for (auto &insts: board_insts) {
            insts.clear();
        }
    }
    void end() override
    {
#pragma omp parallel
#pragma omp single
#pragma omp taskloop
        for (auto &insts: board_insts) {
            std::ranges::stable_sort(insts, [] (auto &a, auto &b) {
                return a.time < b.time;
            });
        }
    }
};

Generator *new_jaqalv1_3_stream_generator()
{
    return new Jaqalv1_3StreamGen;
}

inline int ChannelInfo::add_tone_channel(int chn)
{
    int chn_idx = (int)channels.size();
    channels.push_back({ chn });
    return chn_idx;
}

inline void ChannelInfo::add_seq_channel(int seq_chn, int chn_idx, ToneParam param)
{
    assert(chn_map.count(seq_chn) == 0);
    chn_map.insert({seq_chn, {chn_idx, param}});
}

inline void ChannelInfo::ensure_unused_tones(bool all)
{
    // For now, do not generate RFSoC data if there's no output.
    // This may be a problem if some of the sequences in a scan contains RFSoC outputs
    // while others don't. The artiq integration code would need to handle this case.
    if (channels.empty())
        return;
    // Ensuring both tone being availabe seems to make a difference sometimes.
    // (Could be due to pulse compiler bugs)
    std::bitset<64> tone_used;
    for (auto channel: channels)
        tone_used.set(channel.chn);
    for (int i = 0; i < 32; i++) {
        bool tone0 = tone_used.test(i * 2);
        bool tone1 = tone_used.test(i * 2 + 1);
        if ((tone0 || all) && !tone1) {
            channels.push_back({ i * 2 + 1 });
        }
        if (!tone0 && (tone1 || all)) {
            channels.push_back({ i * 2 });
        }
    }
}

static inline bool parse_action_kws(PyObject *kws, int aid)
{
    assert(kws != Py_None);
    if (!kws)
        return false;
    bool sync = false;
    for (auto [key, value]: pydict_iter(kws)) {
        if (PyUnicode_CompareWithASCIIString(key, "sync") == 0) {
            sync = get_value_bool(value, action_key(aid));
            continue;
        }
        bb_throw_format(PyExc_ValueError, action_key(aid),
                        "Invalid output keyword argument %S", kws);
    }
    return sync;
}

template<typename EventTime>
static __attribute__((always_inline)) inline
void collect_actions(auto *rb, backend::CompiledSeq &cseq, EventTime*)
{
    auto seq = pyx_fld(rb, seq);

    ValueIndexer<int> bool_values;
    ValueIndexer<double> float_values;
    std::vector<Relocation> &relocations = rb->relocations;
    auto event_times = pyx_fld(seq, seqinfo)->time_mgr->event_times;

    rb->channels.ensure_unused_tones(rb->use_all_channels);

    for (auto [seq_chn, value]: rb->channels.chn_map) {
        auto [chn_idx, param] = value;
        auto is_ff = param == ToneFF;
        auto &channel = rb->channels.channels[chn_idx];
        auto &rfsoc_actions = channel.actions[(int)param];
        for (auto action: cseq.all_actions[seq_chn]) {
            auto sync = parse_action_kws(action->kws.get(), action->aid);
            auto value = action->value.get();
            auto is_ramp = py_issubtype_nontrivial(Py_TYPE(value), rampfunctionbase_type);
            if (is_ff && is_ramp)
                bb_throw_format(PyExc_ValueError, action_key(action->aid),
                                "Feed forward control cannot be ramped");
            auto cond = action->cond.get();
            if (cond == Py_False)
                continue;
            bool cond_need_reloc = rtval::is_rtval(cond);
            assert(cond_need_reloc || cond == Py_True);
            int cond_idx = cond_need_reloc ? bool_values.get_id(cond) : -1;
            auto add_action = [&] (auto value, int tid, bool sync, bool is_ramp,
                                   bool is_end) {
                bool needs_reloc = cond_need_reloc;
                Relocation reloc{cond_idx, -1, -1};

                RFSOCAction rfsoc_action;
                rfsoc_action.cond = !cond_need_reloc;
                rfsoc_action.eval_status = false;
                rfsoc_action.isramp = is_ramp;
                rfsoc_action.sync = sync;
                rfsoc_action.aid = action->aid;
                rfsoc_action.tid = tid;
                rfsoc_action.is_end = is_end;

                auto event_time = (EventTime*)PyList_GET_ITEM(event_times, tid);
                if (event_time->data.is_static()) {
                    rfsoc_action.seq_time = event_time->data._get_static();
                }
                else {
                    needs_reloc = true;
                    reloc.time_idx = tid;
                }
                if (is_ramp) {
                    rfsoc_action.ramp = value;
                    auto len = action->length;
                    if (rtval::is_rtval(len)) {
                        needs_reloc = true;
                        reloc.val_idx = float_values.get_id(len);
                    }
                    else {
                        rfsoc_action.float_value =
                            get_value_f64(len, action_key(action->aid));
                    }
                }
                else if (rtval::is_rtval(value)) {
                    needs_reloc = true;
                    if (is_ff) {
                        reloc.val_idx = bool_values.get_id(value);
                    }
                    else {
                        reloc.val_idx = float_values.get_id(value);
                    }
                }
                else if (is_ff) {
                    rfsoc_action.bool_value = get_value_bool(value,
                                                             action_key(action->aid));
                }
                else {
                    rfsoc_action.float_value = get_value_f64(value,
                                                             action_key(action->aid));
                }
                if (needs_reloc) {
                    rfsoc_action.reloc_id = (int)relocations.size();
                    relocations.push_back(reloc);
                }
                else {
                    rfsoc_action.reloc_id = -1;
                }
                rfsoc_actions.push_back(rfsoc_action);
            };
            add_action(action->value.get(), action->tid, sync, is_ramp, false);
            if (action->is_pulse || is_ramp) {
                add_action(action->end_val.get(), action->end_tid, false, false, true);
            }
        }
    }

    rb->bool_values = std::move(bool_values.values);
    rb->float_values = std::move(float_values.values);
}

static constexpr double min_spline_time = 150e-9;

struct SplineBuffer {
    double t[7];
    double v[7];

    bool is_accurate_enough(double threshold) const
    {
        if (t[3] - t[0] <= min_spline_time)
            return true;
        auto sp = spline_from_values(v[0], v[2], v[4], v[6]);
        if (abs(spline_eval(sp, 1.0 / 6) - v[1]) > threshold)
            return false;
        if (abs(spline_eval(sp, 1.0 / 2) - v[3]) > threshold)
            return false;
        if (abs(spline_eval(sp, 5.0 / 6) - v[5]) > threshold)
            return false;
        return true;
    }
};

static __attribute__((flatten))
void _generate_splines(auto &eval_cb, auto &add_sample, SplineBuffer &buff,
                       double threshold)
{
    bb_debug("generate_splines: {%f, %f, %f, %f, %f, %f, %f} -> "
             "{%f, %f, %f, %f, %f, %f, %f}\n",
             buff.t[0], buff.t[1], buff.t[2], buff.t[3],
             buff.t[4], buff.t[5], buff.t[6],
             buff.v[0], buff.v[1], buff.v[2], buff.v[3],
             buff.v[4], buff.v[5], buff.v[6]);
    if (buff.is_accurate_enough(threshold)) {
        bb_debug("accurate enough: t0=%f, t1=%f, t2=%f\n",
                 buff.t[0], buff.t[3], buff.t[6]);
        add_sample(buff.t[3], buff.v[0], buff.v[1], buff.v[2], buff.v[3]);
        add_sample(buff.t[6], buff.v[3], buff.v[4], buff.v[5], buff.v[6]);
        return;
    }
    {
        SplineBuffer buff0;
        {
            double ts[6];
            double vs[6];
            for (int i = 0; i < 6; i++) {
                double t = (buff.t[i] + buff.t[i + 1]) / 2;
                ts[i] = t;
                vs[i] = eval_cb(t);
            }
            bb_debug("evaluate on {%f, %f, %f, %f, %f, %f}\n",
                     ts[0], ts[1], ts[2], ts[3], ts[4], ts[5]);
            buff0 = {
                { buff.t[0], ts[0], buff.t[1], ts[1], buff.t[2], ts[2], buff.t[3] },
                { buff.v[0], vs[0], buff.v[1], vs[1], buff.v[2], vs[2], buff.v[3] },
            };
            buff = {
                { buff.t[3], ts[3], buff.t[4], ts[4], buff.t[5], ts[5], buff.t[6] },
                { buff.v[3], vs[3], buff.v[4], vs[4], buff.v[5], vs[5], buff.v[6] },
            };
        }
        _generate_splines(eval_cb, add_sample, buff0, threshold);
    }
    _generate_splines(eval_cb, add_sample, buff, threshold);
}

static __attribute__((always_inline)) inline
void generate_splines(auto &eval_cb, auto &add_sample, double len, double threshold)
{
    bb_debug("generate_splines: len=%f\n", len);
    SplineBuffer buff;
    for (int i = 0; i < 7; i++) {
        auto t = len * i / 6;
        buff.t[i] = t;
        buff.v[i] = eval_cb(t);
    }
    _generate_splines(eval_cb, add_sample, buff, threshold);
}

inline void
SyncTimeMgr::add_action(std::vector<DDSParamAction> &actions, int64_t start_cycle,
                        int64_t end_cycle, cubic_spline_t sp,
                        int64_t end_seq_time, int tid, ToneParam param)
{
    assert(start_cycle <= end_cycle);
    bb_debug("adding %s spline: [%" PRId64 ", %" PRId64 "], "
             "cycle_len=%" PRId64 ", val=spline(%f, %f, %f, %f)\n",
             param_name(param), start_cycle, end_cycle, end_cycle - start_cycle,
             sp.order0, sp.order1, sp.order2, sp.order3);
    auto has_sync = [&] {
        return next_it != times.end() && next_it->second.seq_time <= end_seq_time;
    };
    if (param != ToneFreq || !has_sync()) {
        if (param == ToneFreq)
            bb_debug("  No sync to handle: last_sync: %d, time: %" PRId64
                     ", sync_time: %" PRId64 "\n",
                     next_it == times.end(), end_seq_time,
                     next_it == times.end() ? -1 : next_it->second.seq_time);
        if (end_cycle != start_cycle)
            actions.push_back({ end_cycle - start_cycle, false, sp });
        return;
    }
    auto sync_cycle = next_it->first;
    auto sync_info = next_it->second;
    // First check if we need to update the sync frequency,
    // If there are multiple frequency values at exactly the same sequence time
    // we pick the last one that we see unless there's a frequency action
    // at exactly the same time point (same tid) as the sync action.
    assert(sync_freq_seq_time <= sync_info.seq_time);
    bb_debug("  sync_time: %" PRId64 ", sync_tid: %d, "
             "sync_freq_time: %" PRId64 ", sync_freq_match_tid: %d\n",
             sync_info.seq_time, sync_info.tid, sync_freq_seq_time,
             sync_freq_match_tid);
    if (sync_freq_seq_time < sync_info.seq_time || !sync_freq_match_tid) {
        sync_freq_seq_time = sync_info.seq_time;
        sync_freq_match_tid = sync_info.tid == tid;

        if (sync_cycle == start_cycle) {
            sync_freq = sp.order0;
        }
        else if (sync_cycle == end_cycle) {
            sync_freq = sp.order0 + sp.order1 + sp.order2 + sp.order3;
        }
        else {
            auto t = double(sync_cycle - start_cycle) / double(end_cycle - start_cycle);
            sync_freq = spline_eval(sp, t);
        }
        bb_debug("  updated sync frequency: %f @%" PRId64 ", sync_freq_match_tid: %d\n",
                 sync_freq, sync_freq_seq_time, sync_freq_match_tid);
    }
    assert(sync_cycle <= end_cycle);
    assert(sync_cycle >= start_cycle);

    if (sync_cycle == end_cycle) {
        // Sync at the end of the spline, worry about it next time.
        bb_debug("  sync at end, skip until next one @%" PRId64 "\n", end_cycle);
        if (end_cycle != start_cycle)
            actions.push_back({ end_cycle - start_cycle, false, sp });
        return;
    }

    assert(end_cycle > start_cycle);
    bool need_sync = true;
    if (sync_cycle > start_cycle) {
        bb_debug("  Output until @%" PRId64 "\n", sync_cycle);
        actions.push_back({ sync_cycle - start_cycle, false,
                spline_resample_cycle(sp, start_cycle, end_cycle,
                                      start_cycle, sync_cycle) });
    } else if (sync_freq != sp.order0) {
        // We have a sync at frequency action boundary.
        // This is the only case we may need to sync at a different frequency
        // compared to the frequency of the output immediately follows this.
        bb_debug("  0-length sync @%" PRId64 "\n", start_cycle);
        actions.push_back({ 0, true, spline_from_static(sync_freq) });
        need_sync = false;
    }
    while (true) {
        // Status:
        // * output is at `sync_cycle`
        // * `next_it` points to a sync event at `sync_cycle`
        // * `need_sync` records whether the next action needs sync'ing
        // * `sync_cycle < end_cycle`

        assert(end_cycle > sync_cycle);

        // First figure out what's the end of the current time segment.
        ++next_it;
        if (!has_sync()) {
            bb_debug("  Reached end of spline: sync=%d\n", need_sync);
            actions.push_back({ end_cycle - sync_cycle, need_sync,
                    spline_resample_cycle(sp, start_cycle, end_cycle,
                                          sync_cycle, end_cycle) });
            return;
        }
        // If we have another sync to handle, do the output
        // and compute the new sync frequency
        auto prev_cycle = sync_cycle;
        sync_cycle = next_it->first;
        assert(sync_cycle > prev_cycle);
        assert(sync_cycle <= end_cycle);
        bb_debug("  Output until @%" PRId64 ", sync=%d\n", sync_cycle, need_sync);
        actions.push_back({ sync_cycle - prev_cycle, need_sync,
                spline_resample_cycle(sp, start_cycle, end_cycle,
                                      prev_cycle, sync_cycle) });
        need_sync = true;
        sync_info = next_it->second;
        sync_freq_seq_time = sync_info.seq_time;
        assert(sync_info.tid != tid);
        sync_freq_match_tid = false;
        if (sync_cycle == end_cycle) {
            sync_freq = sp.order0 + sp.order1 + sp.order2 + sp.order3;
            bb_debug("  updated sync frequency: %f @%" PRId64 ", sync_freq_match_tid: %d\n"
                     "  sync at end, skip until next one @%" PRId64 "\n",
                     sync_freq, sync_freq_seq_time, sync_freq_match_tid, end_cycle);
            return;
        }
        else {
            auto t = double(sync_cycle - start_cycle) / double(end_cycle - start_cycle);
            sync_freq = spline_eval(sp, t);
        }
        bb_debug("  updated sync frequency: %f @%" PRId64 ", sync_freq_match_tid: %d\n",
                 sync_freq, sync_freq_seq_time, sync_freq_match_tid);
    }
}

template<typename _RampFunctionBase, typename SeqCubicSpline>
static __attribute__((always_inline)) inline
void gen_rfsoc_data(auto *rb, backend::CompiledSeq &cseq,
                    _RampFunctionBase*, SeqCubicSpline*)
{
    bb_debug("gen_rfsoc_data: start\n");
    auto seq = pyx_fld(rb, seq);
    for (size_t i = 0, nreloc = rb->bool_values.size(); i < nreloc; i++) {
        auto &[rtval, val] = rb->bool_values[i];
        val = !rtval::rtval_cache((rtval::_RuntimeValue*)rtval).is_zero();
    }
    for (size_t i = 0, nreloc = rb->float_values.size(); i < nreloc; i++) {
        auto &[rtval, val] = rb->float_values[i];
        val = rtval::rtval_cache((rtval::_RuntimeValue*)rtval).template get<double>();
    }
    auto &time_values = pyx_fld(seq, seqinfo)->time_mgr->time_values;
    auto reloc_action = [rb, &time_values] (const RFSOCAction &action,
                                            ToneParam param) {
        auto reloc = rb->relocations[action.reloc_id];
        if (reloc.cond_idx != -1)
            action.cond = rb->bool_values[reloc.cond_idx].second;
        // No need to do anything else if we hit a disabled action.
        if (!action.cond)
            return;
        if (reloc.time_idx != -1)
            action.seq_time = time_values[reloc.time_idx];
        if (reloc.val_idx != -1) {
            if (param == ToneFF) {
                action.bool_value = rb->bool_values[reloc.val_idx].second;
            }
            else {
                action.float_value = rb->float_values[reloc.val_idx].second;
            }
        }
    };

    bool eval_status = !rb->eval_status;
    rb->eval_status = eval_status;

    constexpr double spline_threshold[3] = {
        [ToneFreq] = 10,
        [TonePhase] = 1e-3,
        [ToneAmp] = 1e-3,
    };

    int64_t max_delay = 0;
    for (auto [dds, delay]: rb->channels.dds_delay)
        max_delay = std::max(max_delay, delay);

    auto reloc_and_cmp_action = [&] (const auto &a1, const auto &a2, auto param) {
        if (a1.reloc_id >= 0 && a1.eval_status != eval_status) {
            a1.eval_status = eval_status;
            reloc_action(a1, (ToneParam)param);
        }
        if (a2.reloc_id >= 0 && a2.eval_status != eval_status) {
            a2.eval_status = eval_status;
            reloc_action(a2, (ToneParam)param);
        }
        // Move disabled actions to the end
        if (a1.cond != a2.cond)
            return int(a1.cond) > int(a2.cond);
        // Sort by time
        if (a1.seq_time != a2.seq_time)
            return a1.seq_time < a2.seq_time;
        // Sometimes time points with different tid needs
        // to be sorted by tid to get the output correct.
        if (a1.tid != a2.tid)
            return a1.tid < a2.tid;
        // End action technically happens
        // just before the time point and must be sorted
        // to be before the start action.
        return int(a1.is_end) > int(a2.is_end);
        // The frontend/shared finalization code only allow
        // a single action on the same channel at the same time
        // so there shouldn't be any ambiguity.
    };

    auto reloc_sort_actions = [&] (auto &actions, auto param) {
        if (actions.size() == 1) {
            auto &a = actions[0];
            if (a.reloc_id >= 0) {
                a.eval_status = eval_status;
                reloc_action(a, (ToneParam)param);
            }
        }
        else {
            std::ranges::sort(actions, [&] (const auto &a1, const auto &a2) {
                return reloc_and_cmp_action(a1, a2, param);
            });
        }
    };

    auto gen = rb->generator->gen.get();
    gen->start();

    // Add extra cycles to be able to handle the requirement of minimum 4 cycles.
    auto total_cycle = seq_time_to_cycle(cseq.total_time + max_delay) + 8;
    for (auto &channel: rb->channels.channels) {
        ScopeExit cleanup([&] {
            rb->tone_buffer.clear();
        });
        int64_t dds_delay = 0;
        if (auto it = rb->channels.dds_delay.find(channel.chn >> 1);
            it != rb->channels.dds_delay.end())
            dds_delay = it->second;
        auto sync_mgr = rb->tone_buffer.syncs;
        {
            auto &actions = channel.actions[ToneFF];
            bb_debug("processing tone channel: %d, ff, nactions=%zd\n",
                     channel.chn, actions.size());
            reloc_sort_actions(actions, ToneFF);
            int64_t cur_cycle = 0;
            bool ff = false;
            auto &ff_action = rb->tone_buffer.ff;
            assert(ff_action.empty());
            for (auto &action: actions) {
                if (!action.cond) {
                    bb_debug("found disabled ff action, finishing\n");
                    break;
                }
                auto action_seq_time = action.seq_time + dds_delay;
                auto new_cycle = seq_time_to_cycle(action_seq_time);
                if (action.sync)
                    sync_mgr.add(action_seq_time, new_cycle, action.tid, ToneFF);
                // Nothing changed.
                if (ff == action.bool_value) {
                    bb_debug("skipping ff action: @%" PRId64 ", ff=%d\n",
                             new_cycle, ff);
                    continue;
                }
                if (new_cycle != cur_cycle) {
                    bb_debug("adding ff action: [%" PRId64 ", %" PRId64 "], "
                             "cycle_len=%" PRId64 ", ff=%d\n",
                             cur_cycle, new_cycle, new_cycle - cur_cycle, ff);
                    assert(new_cycle > cur_cycle);
                    ff_action.push_back({ new_cycle - cur_cycle, ff });
                    cur_cycle = new_cycle;
                }
                ff = action.bool_value;
                bb_debug("ff status: @%" PRId64 ", ff=%d\n", cur_cycle, ff);
            }
            bb_debug("adding last ff action: [%" PRId64 ", %" PRId64 "], "
                     "cycle_len=%" PRId64 ", ff=%d\n", cur_cycle,
                     total_cycle, total_cycle - cur_cycle, ff);
            assert(total_cycle > cur_cycle);
            ff_action.push_back({ total_cycle - cur_cycle, ff });
        }
        for (auto param: { ToneAmp, TonePhase, ToneFreq }) {
            auto &actions = channel.actions[param];
            sync_mgr.init_output(param);
            bb_debug("processing tone channel: %d, %s, nactions=%zd\n",
                     channel.chn, param_name(param), actions.size());
            reloc_sort_actions(actions, param);
            int64_t cur_cycle = 0;
            auto &param_action = rb->tone_buffer.params[param];
            assert(param_action.empty());
            double val = 0;
            int prev_tid = -1;
            for (auto &action: actions) {
                if (!action.cond) {
                    bb_debug("found disabled %s action, finishing\n",
                             param_name(param));
                    break;
                }
                auto action_seq_time = action.seq_time + dds_delay;
                auto new_cycle = seq_time_to_cycle(action_seq_time);
                if (action.sync)
                    sync_mgr.add(action_seq_time, new_cycle, action.tid, param);
                if (!action.isramp && val == action.float_value) {
                    bb_debug("skipping %s action: @%" PRId64 ", val=%f\n",
                             param_name(param), new_cycle, val);
                    continue;
                }
                sync_mgr.add_action(param_action, cur_cycle, new_cycle,
                                    spline_from_static(val), action_seq_time,
                                    prev_tid, param);
                cur_cycle = new_cycle;
                prev_tid = action.tid;
                if (!action.isramp) {
                    val = action.float_value;
                    bb_debug("%s status: @%" PRId64 ", val=%f\n",
                             param_name(param), cur_cycle, val);
                    continue;
                }
                auto len = action.float_value;
                auto ramp_func = (_RampFunctionBase*)action.ramp;
                bb_debug("processing ramp on %s: @%" PRId64 ", len=%f, func=%p\n",
                         param_name(param), cur_cycle, len, ramp_func);
                double sp_time;
                int64_t sp_seq_time;
                int64_t sp_cycle;
                auto update_sp_time = [&] (double t) {
                    static constexpr double time_scale = event_time::time_scale;
                    sp_time = t;
                    sp_seq_time = action_seq_time + int64_t(t * time_scale + 0.5);
                    sp_cycle = seq_time_to_cycle(sp_seq_time);
                };
                update_sp_time(0);
                auto add_spline = [&] (double t2, cubic_spline_t sp) {
                    assert(t2 >= sp_time);
                    auto cycle1 = sp_cycle;
                    update_sp_time(t2);
                    auto cycle2 = sp_cycle;
                    // The spline may not actually start on the cycle.
                    // However, attempting to resample the spline results in
                    // more unique splines being created which seems to be overflowing
                    // the buffer on the hardware.
                    sync_mgr.add_action(param_action, cycle1, cycle2,
                                        sp, sp_seq_time, prev_tid, param);
                };
                if (Py_TYPE(ramp_func) == (PyTypeObject*)seqcubicspline_type) {
                    bb_debug("found SeqCubicSpline on %s spline: "
                             "old cycle:%" PRId64 "\n", param_name(param), cur_cycle);
                    auto py_spline = (SeqCubicSpline*)ramp_func;
                    cubic_spline_t sp{py_spline->f_order0, py_spline->f_order1,
                        py_spline->f_order2, py_spline->f_order3};
                    val = sp.order0 + sp.order1 + sp.order2 + sp.order3;
                    add_spline(len, sp);
                    cur_cycle = sp_cycle;
                    bb_debug("found SeqCubicSpline on %s spline: "
                             "new cycle:%" PRId64 "\n", param_name(param), cur_cycle);
                    continue;
                }
                auto add_sample = [&] (double t2, double v0, double v1,
                                       double v2, double v3) {
                    add_spline(t2, spline_from_values(v0, v1, v2, v3));
                    val = v3;
                };
                auto _runtime_eval = ramp_func->__pyx_vtab->runtime_eval;
                auto eval_ramp = [&] (double t) {
                    auto v = _runtime_eval(ramp_func, t);
                    throw_py_error(v.err);
                    return v.val.f64_val;
                };
                py_object pts(ramp_func->__pyx_vtab->spline_segments(ramp_func, len, val));
                bb_reraise_and_throw_if(!pts, action_key(action.aid));
                if (pts == Py_None) {
                    bb_debug("Use adaptive segments on %s spline: "
                             "old cycle:%" PRId64 "\n", param_name(param), cur_cycle);
                    generate_splines(eval_ramp, add_sample, len,
                                     spline_threshold[param]);
                    cur_cycle = sp_cycle;
                    bb_debug("Use adaptive segments on %s spline: "
                             "new cycle:%" PRId64 "\n", param_name(param), cur_cycle);
                    continue;
                }
                py_object iter(PyObject_GetIter(pts));
                bb_reraise_and_throw_if(!iter, action_key(action.aid));
                double prev_t = 0;
                double prev_v = eval_ramp(0);
                bb_debug("Use ramp function provided segments on %s spline: "
                         "old cycle:%" PRId64 "\n", param_name(param), cur_cycle);
                while (PyObject *_item = PyIter_Next(iter.get())) {
                    py_object item(_item);
                    double t = PyFloat_AsDouble(item.get());
                    if (!(t > prev_t)) [[unlikely]] {
                        if (!PyErr_Occurred()) {
                            if (t < 0) {
                                PyErr_Format(PyExc_ValueError,
                                             "Segment time cannot be negative");
                            }
                            else {
                                PyErr_Format(PyExc_ValueError,
                                             "Segment time point must "
                                             "monotonically increase");
                            }
                        }
                        bb_rethrow(action_key(action.aid));
                    }
                    auto t1 = t * (1.0 / 3.0) + prev_t * (2.0 / 3.0);
                    auto t2 = t * (2.0 / 3.0) + prev_t * (1.0 / 3.0);
                    auto t3 = t;
                    auto v0 = prev_v;
                    auto v1 = eval_ramp(t1);
                    auto v2 = eval_ramp(t2);
                    auto v3 = eval_ramp(t3);
                    add_sample(t3, v0, v1, v2, v3);
                    prev_t = t3;
                    prev_v = v3;
                }
                bb_reraise_and_throw_if(PyErr_Occurred(), action_key(action.aid));
                if (!(prev_t < len)) [[unlikely]]
                    bb_throw_format(PyExc_ValueError, action_key(action.aid),
                                    "Segment time point must not "
                                    "exceed action length.");
                auto t1 = len * (1.0 / 3.0) + prev_t * (2.0 / 3.0);
                auto t2 = len * (2.0 / 3.0) + prev_t * (1.0 / 3.0);
                auto t3 = len;
                auto v0 = prev_v;
                auto v1 = eval_ramp(t1);
                auto v2 = eval_ramp(t2);
                auto v3 = eval_ramp(t3);
                add_sample(t3, v0, v1, v2, v3);
                cur_cycle = sp_cycle;
                bb_debug("Use ramp function provided segments on %s spline: "
                         "new cycle:%" PRId64 "\n", param_name(param), cur_cycle);
            }
            sync_mgr.add_action(param_action, cur_cycle, total_cycle,
                                spline_from_static(val), cycle_to_seq_time(total_cycle),
                                prev_tid, param);
        }
        gen->process_channel(rb->tone_buffer, channel.chn, total_cycle);
    }
    gen->end();
    bb_debug("gen_rfsoc_data: finish\n");
}

}
