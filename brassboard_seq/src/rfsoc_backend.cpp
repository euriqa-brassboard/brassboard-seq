/*************************************************************************
 *   Copyright (c) 2024 - 2024 Yichao Yu <yyc1992@gmail.com>             *
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

#include <bitset>

#include <assert.h>

namespace brassboard_seq::rfsoc_backend {

static PyTypeObject *rampfunction_type;
static PyTypeObject *seqcubicspline_type;

// order of the coefficient is order0, order1, order2, order3
template<int nbits> static constexpr inline std::pair<std::array<int64_t,4>,int>
convert_pdq_spline(std::array<double,4> sp, int64_t cycles, double scale)
{
    static_assert(nbits < 64);
    double tstep = 1 / double(cycles);
    std::array<int64_t,4> isp;
    double tstep2 = tstep * tstep;
    double tstep3 = tstep2 * tstep;
    sp[1] = tstep * (sp[1] + sp[2] * tstep + sp[3] * tstep2);
    sp[2] = 2 * tstep2 * (sp[2] + 3 * sp[3] * tstep);
    sp[3] = 6 * tstep3 * sp[3];

    constexpr uint64_t mask = (uint64_t(1) << nbits) - 1;
    constexpr double bitscale = double(uint64_t(1) << nbits);
    // See below, a threshold of `1 - 2^(-nbits)` should already be sufficient
    // to make sure the value doesn't round up.
    // Here we use `1 - 2 * 2^(-nbits)` just to be safe in case
    // we got the rounding mode wrong or sth...
    // FIXME: constexpr in C++23
    double round_thresh = 1 - 2 * std::ldexp(1, -nbits);

    // Now we'll map floating point values [-0.5, 0.5] to [-2^(nbits-1), 2^(nbits-1)]

    // For the 0-th order, we can just round the number
    isp[0] = std::llrint(sp[0] * (scale * bitscale)) & mask;

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

    isp[1] = std::llrint(std::ldexp(sp[1], _exp[0] + nbits + shift_len)) & mask;
    isp[2] = std::llrint(std::ldexp(sp[2], _exp[1] + nbits + shift_len * 2)) & mask;
    isp[3] = std::llrint(std::ldexp(sp[3], _exp[2] + nbits + shift_len * 3)) & mask;
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
        JaqalInst pulse;
        pulse |= JaqalInst(metadata) << Bits::METADATA;
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
        pulse |= JaqalInst(isp[0]) << 40 * 0;
        pulse |= JaqalInst(isp[1]) << 40 * 1;
        pulse |= JaqalInst(isp[2]) << 40 * 2;
        pulse |= JaqalInst(isp[3]) << 40 * 3;
        pulse |= JaqalInst(cycles) << 40 * 4;
        return pulse;
    }

    // certain metadata bits should only be applied
    // with the first pulse such as waittrig and sync
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
    static constexpr inline auto
    freq_pulse(int channel, int tone, cubic_spline_t sp, int64_t cycles, bool waittrig,
               bool sync, bool fb_enable)
    {
        assert(cycles >= 4);
        assert((cycles >> 40) == 0);
        auto [isp, shift_len] = convert_pdq_spline_freq(sp.to_array(), cycles);
        auto metadata = raw_param_metadata(tone ? ModType::FRQMOD1 : ModType::FRQMOD0,
                                           channel, shift_len, waittrig,
                                           sync, false, fb_enable);
        return pulse(metadata, isp, cycles);
    }
    static constexpr inline auto
    amp_pulse(int channel, int tone, cubic_spline_t sp, int64_t cycles, bool waittrig,
              bool sync=false, bool fb_enable=false)
    {
        assert(cycles >= 4);
        assert((cycles >> 40) == 0);
        auto [isp, shift_len] = convert_pdq_spline_amp(sp.to_array(), cycles);
        auto metadata = raw_param_metadata(tone ? ModType::AMPMOD1 : ModType::AMPMOD0,
                                           channel, shift_len, waittrig,
                                           sync, false, fb_enable);
        return pulse(metadata, isp, cycles);
    }
    static constexpr inline auto
    phase_pulse(int channel, int tone, cubic_spline_t sp, int64_t cycles, bool waittrig,
                bool sync=false, bool fb_enable=false)
    {
        assert(cycles >= 4);
        assert((cycles >> 40) == 0);
        auto [isp, shift_len] = convert_pdq_spline_phase(sp.to_array(), cycles);
        auto metadata = raw_param_metadata(tone ? ModType::PHSMOD1 : ModType::PHSMOD0,
                                           channel, shift_len, waittrig,
                                           sync, false, fb_enable);
        return pulse(metadata, isp, cycles);
    }

    // certain metadata bits should only be applied
    // with the first pulse such as waittrig and rst_frame
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
    static constexpr inline auto
    frame_pulse(int channel, int tone, cubic_spline_t sp, int64_t cycles,
                bool waittrig, bool apply_at_end, bool rst_frame,
                int fwd_frame_mask, int inv_frame_mask)
    {
        assert(cycles >= 4);
        assert((cycles >> 40) == 0);
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
        struct StrKey {
            PyObject *str;
            Py_hash_t hash;
            StrKey(PyObject *str)
                : str(str),
                  hash(PyObject_Hash(str))
            {}
        };
        PyObject *channel_list[64];
        PyObject *CubicSpline;
        PyObject *ToneData;
        PyObject *cubic_0;
        std::vector<std::pair<StrKey,PyObject*>> tonedata_fields;
        StrKey channel_key;
        StrKey tone_key;
        StrKey duration_cycles_key;
        StrKey frequency_hz_key;
        StrKey amplitude_key;
        StrKey phase_rad_key;
        StrKey frame_rotation_rad_key;
        StrKey wait_trigger_key;
        StrKey sync_key;
        StrKey output_enable_key;
        StrKey feedback_enable_key;
        StrKey bypass_lookup_tables_key;

        static inline void dict_setitem(PyObject *dict, StrKey key, PyObject *value)
        {
            throw_if(_PyDict_SetItem_KnownHash(dict, key.str, value, key.hash));
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
    ~PulseCompilerGen() override
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

    PyObject *key, *value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(td_dict, &pos, &key, &value)) {
        for (auto name: {"channel", "tone", "duration_cycles", "frequency_hz",
                "amplitude", "phase_rad", "frame_rotation_rad", "wait_trigger",
                "sync", "output_enable", "feedback_enable",
                "bypass_lookup_tables"}) {
            if (PyUnicode_CompareWithASCIIString(key, name) == 0) {
                goto skip_field;
            }
        }
        tonedata_fields.push_back({ { py_newref(key) } , py_newref(value) });
    skip_field:
        ;
    }
}

void SyncChannelGen::process_channel(ToneBuffer &tone_buffer, int chn,
                                     int64_t total_cycle)
{
    bool first_output = true;
    auto get_trigger = [&] {
        auto v = first_output;
        first_output = false;
        return v;
    };
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
                          { get_trigger(), sync, ff_action.ff }, cur_cycle);
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
                          { get_trigger(), sync, ff_action.ff }, cur_cycle);
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

inline void ChannelInfo::ensure_both_tones()
{
    // Ensuring both tone being availabe seems to make a difference sometimes.
    // (Could be due to pulse compiler bugs)
    std::bitset<64> tone_used;
    for (auto channel: channels)
        tone_used.set(channel.chn);
    for (int i = 0; i < 32; i++) {
        bool tone0 = tone_used.test(i * 2);
        bool tone1 = tone_used.test(i * 2 + 1);
        if (tone0 && !tone1) {
            channels.push_back({ i * 2 + 1 });
        }
        else if (!tone0 && tone1) {
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
    PyObject *key, *value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(kws, &pos, &key, &value)) {
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
void collect_actions(auto *rb, EventTime*)
{
    auto seq = rb->__pyx_base.seq;

    ValueIndexer<int> bool_values;
    ValueIndexer<double> float_values;
    std::vector<Relocation> &relocations = rb->relocations;
    auto event_times = seq->__pyx_base.__pyx_base.seqinfo->time_mgr->event_times;

    rb->channels.ensure_both_tones();

    for (auto [seq_chn, value]: rb->channels.chn_map) {
        auto [chn_idx, param] = value;
        auto is_ff = param == ToneFF;
        auto &channel = rb->channels.channels[chn_idx];
        auto &rfsoc_actions = channel.actions[(int)param];
        for (auto action: seq->all_actions[seq_chn]) {
            auto sync = parse_action_kws(action->kws.get(), action->aid);
            auto value = action->value.get();
            auto is_ramp = py_issubtype_nontrivial(Py_TYPE(value), rampfunction_type);
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

static inline cubic_spline_t
spline_resample_cycle(cubic_spline_t sp, int64_t start, int64_t end,
                      int64_t cycle1, int64_t cycle2)
{
    if (cycle1 == start && cycle2 == end)
        return sp;
    return spline_resample(sp, double(cycle1 - start) / double(end - start),
                           double(cycle2 - start) / double(end - start));
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

template<typename RampFunction, typename SeqCubicSpline>
static __attribute__((always_inline)) inline
void gen_rfsoc_data(auto *rb, RampFunction*, SeqCubicSpline*)
{
    bb_debug("gen_rfsoc_data: start\n");
    auto seq = rb->__pyx_base.seq;
    for (size_t i = 0, nreloc = rb->bool_values.size(); i < nreloc; i++) {
        auto &[rtval, val] = rb->bool_values[i];
        val = !rtval::rtval_cache((rtval::_RuntimeValue*)rtval).is_zero();
    }
    for (size_t i = 0, nreloc = rb->float_values.size(); i < nreloc; i++) {
        auto &[rtval, val] = rb->float_values[i];
        val = rtval::rtval_cache((rtval::_RuntimeValue*)rtval).template get<double>();
    }
    auto &time_values = seq->__pyx_base.__pyx_base.seqinfo->time_mgr->time_values;
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

    // Add extra cycles to be able to handle the requirement of minimum 4 cycles.
    auto total_cycle = seq_time_to_cycle(rb->__pyx_base.seq->total_time + max_delay) + 8;
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
                auto ramp_func = (RampFunction*)action.ramp;
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
                if (Py_TYPE(ramp_func) == seqcubicspline_type) {
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
    bb_debug("gen_rfsoc_data: finish\n");
}

}
