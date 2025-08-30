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

#ifndef BRASSBOARD_SEQ_SRC_RFSOC_H
#define BRASSBOARD_SEQ_SRC_RFSOC_H

#include "utils.h"

#include <array>

namespace brassboard_seq::rfsoc {

static constexpr inline int64_t bitcast_f64_i64(double f)
{
    static_assert(sizeof(double) == 8);
    union {
        double f;
        int64_t i;
    } u{ .f = f };
    return u.i;
}

struct F64Parts {
    int exp;
    int64_t frac;
    constexpr F64Parts(double f)
    {
        int64_t fi = bitcast_f64_i64(f);
        int64_t sign = fi >> 63;
        exp = int(fi >> 52) & ((1 << 11) - 1);
        // This does not have any special treatment for 0, subnormal numbers, or nan/inf.
        // However, we don't need to handle nan/inf in any particular way
        // and the conversion to spline coefficient will treat small numbers,
        // including 0 and subnormal numbers, as zero anyway so we don't need to be
        // accurate here. This is saves an instruction or two.
        frac = (fi & ((int64_t(1) << 52) - 1)) | (int64_t(1) << 52);
        frac = (sign + frac) ^ sign;
    }
};

static constexpr inline __attribute__((always_inline)) std::pair<std::array<int64_t,4>,int>
encode_pdq_spline(int64_t isp0, double sp1, double sp2, double sp3)
{
    constexpr uint64_t mask = (uint64_t(1) << 40) - 1;

    std::array<int64_t,4> isp;
    isp[0] = isp0;

    // Cross over to integer domain to figure out the shift value to use.
    // For each of the f64 parts, the exp is a number in [0, 2047]
    // and frac is a 54-bit signed integer that is signed extended to 64 bits.
    F64Parts fparts[] = {sp1, sp2, sp3};

    // First compute the shift for the spline.
    // Since the spline coefficients for 1st, 2nd and 3rd orders
    // are only precise to (2^0, 2^-16, 2^-32) respectively, a shift of at most 11
    // (which provides a precision of (2^-11, 2^-22, 2^-33) precision for the 3 orders)
    // is sufficient to provide the full precision of the spline engine hardware.
    int shift_len = 11;

    // The coefficients have been scaled to [-0.5, 0.5] signed,
    // or equivalently [0, 1] unsigned.
    // The f64parts for each number contains 54 significant digits.
    // Bit 0 to bit 53 for each of the `frac` corresponds to
    // `2^(exp - 1075)` to `2^(exp - 1022)`.
    // Bit 0 to bit 39 for each of the spline coefficient corresponds to
    // `2^(-40 - shift_len * order)` to `2^(-1 - shift_len * order)`.
    // In order for us to not loose the top bits, we need
    // `(-1 - shift_len * order) >= (exp - 1022)` or
    // `shift_len <= (1021 - exp) / order`
    for (int i = 0; i < 3; i++)
        shift_len = std::min(shift_len, (1021 - fparts[i].exp) / (i + 1));
    if (shift_len < 0) [[unlikely]] // overflowing value
        shift_len = 0;

    auto try_shift_len = [&] (bool overflow_check) {
        // Now we need to round each value and shift them to the right place.
        // The hardware precision for each order is `2^(-40 - 16 * (order - 1))`
        // which corresponds to bit `1035 - 16 * (order - 1) - exp` in `frac`.
        // However, limited by `shift_len`, the percision is also affected by
        // how we encode the coefficients in the spline which limits it to
        // `2^(-40 - shift_len * order)` which is bit `1035 - shift_len * order - exp`
        // in `frac`.
        for (int i = 0; i < 3; i++) {
            int order = i + 1;
            // This is the precision bit for this coefficient.
            int bit = 1035 - std::min(shift_len * order, 16 * i) - fparts[i].exp;
            if (bit > 54) {
                isp[order] = 0;
            }
            else {
                int64_t frac = fparts[i].frac;
                int shiftbit = 1035 - shift_len * order - fparts[i].exp;
                assert(shiftbit <= bit);
                if (shiftbit < 14) [[unlikely]] {
                    // overflow, do whatever,
                    isp[order] = 0;
                }
                else {
                    assert(bit > 0);
                    auto newfrac = frac + (int64_t(1) << (bit - 1));
                    auto overflow = (shiftbit == 14 && shift_len > 0 &&
                                     (newfrac & ~frac & (int64_t(1) << 53)));
                    // Overflow from rounding, try again with a smaller shift
                    if (overflow_check && overflow) [[unlikely]] {
                        shift_len--;
                        return false;
                    }
                    assert(!overflow);
                    isp[order] = (newfrac >> shiftbit) & mask;
                }
            }
        }
        return true;
    };

    if (!try_shift_len(true)) [[unlikely]]
        try_shift_len(false);
    return { isp, shift_len };
}

// order of the coefficient is order0, order1, order2, order3
static constexpr inline __attribute__((flatten,always_inline))
std::pair<std::array<int64_t,4>,int>
convert_pdq_spline(std::array<double,4> sp, int64_t cycles, double scale)
{
    constexpr uint64_t mask = (uint64_t(1) << 40) - 1;
    constexpr double bitscale = double(uint64_t(1) << 40);

    // For the 0-th order, we can just round the number
    int64_t isp0 = (__builtin_constant_p(sp[0]) && sp[0] == 0) ? 0 :
        round<int64_t>(sp[0] * (scale * bitscale)) & mask;
    if (sp[1] == 0 && sp[2] == 0 && sp[3] == 0)
        return {{isp0, 0, 0, 0}, 0};

    // The spline engine uses the coefficients in an accumulator fashion.
    // In particular, the four orders (order0 - order3) are updated on each cycle
    // using `order{n} <= order{n} + order{n + 1}`
    // and the `order0` is used as the output of the spline engine.
    // The following converts the input polynomial coefficients
    // to the accumulator style coefficient and
    // normalize all of them so a full scale value corresponds to `1.0`.
    double tstep = 1 / double(cycles);
    double tstep2 = tstep * tstep;
    double tstep3 = tstep2 * tstep;
    sp[1] = (tstep * (sp[1] + sp[2] * tstep + sp[3] * tstep2)) * scale;
    sp[2] = (2 * tstep2 * (sp[2] + 3 * sp[3] * tstep)) * scale;
    sp[3] = (6 * tstep3 * sp[3]) * scale;

    return encode_pdq_spline(isp0, sp[1], sp[2], sp[3]);
}

constexpr static double output_clock = 819.2e6;

static constexpr inline std::pair<std::array<int64_t,4>,int>
convert_pdq_spline_freq(std::array<double,4> sp, int64_t cycles)
{
    return convert_pdq_spline(sp, cycles, 1 / output_clock);
}

static constexpr inline std::pair<std::array<int64_t,4>,int>
convert_pdq_spline_amp(std::array<double,4> sp, int64_t cycles)
{
    auto [isp, shift_len] = convert_pdq_spline(sp, cycles, 0.5 - 0x1p-17);
    return { isp, shift_len };
}

static constexpr inline std::pair<std::array<int64_t,4>,int>
convert_pdq_spline_phase(std::array<double,4> sp, int64_t cycles)
{
    return convert_pdq_spline(sp, cycles, 1);
}

using JaqalInst = Bits<int64_t,4>;

struct TimedID {
    int64_t time;
    int16_t id;
};

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

// Set the minimum clock cycles for a pulse to help avoid underflows. This time
// is determined by state machine transitions for loading another gate, but does
// not account for serialization of pulse words.
static constexpr int MINIMUM_PULSE_CLOCK_CYCLES = 4;

namespace Jaqal_v1 {
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
freq_pulse(int channel, int tone, cubic_spline sp, int64_t cycles, bool waittrig,
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
amp_pulse(int channel, int tone, cubic_spline sp, int64_t cycles, bool waittrig,
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
phase_pulse(int channel, int tone, cubic_spline sp, int64_t cycles, bool waittrig,
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
frame_pulse(int channel, int tone, cubic_spline sp, int64_t cycles,
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

    void end();
};

} // Jaqal_v1

namespace Jaqal_v1_3 {
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

static constexpr inline __attribute__((always_inline,flatten)) auto
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

static constexpr auto modtype_mask = JaqalInst::mask(Bits::MODTYPE, Bits::MODTYPE + 7);
static constexpr auto channel_mask = JaqalInst::mask(Bits::DMA_MUX, Bits::DMA_MUX + 7);
static constexpr auto modtype_nmask = ~modtype_mask;
static constexpr auto channel_nmask = ~channel_mask;
static constexpr inline __attribute__((always_inline,flatten)) auto
apply_modtype_mask(JaqalInst pulse, ModTypeMask mod_mask)
{
    return (pulse & modtype_nmask) | JaqalInst(uint8_t(mod_mask)) << Bits::MODTYPE;
}
static constexpr inline __attribute__((always_inline,flatten)) auto
apply_channel_mask(JaqalInst pulse, uint8_t chn_mask)
{
    return (pulse & channel_nmask) | JaqalInst(chn_mask) << Bits::DMA_MUX;
}

static constexpr inline __attribute__((always_inline,flatten)) uint64_t
raw_param_metadata(int shift_len, bool waittrig, bool sync, bool enable, bool fb_enable)
{
    assert(shift_len >= 0 && shift_len < 32);
    uint64_t metadata = uint64_t(shift_len) << (Bits::SPLSHIFT - Bits::METADATA);
    metadata |= uint64_t(waittrig) << (Bits::WAIT_TRIG - Bits::METADATA);
    metadata |= uint64_t(enable) << (Bits::OUTPUT_EN - Bits::METADATA);
    metadata |= uint64_t(fb_enable) << (Bits::FRQ_FB_EN - Bits::METADATA);
    metadata |= uint64_t(sync) << (Bits::SYNC_FLAG - Bits::METADATA);
    return metadata;
}
static constexpr inline __attribute__((always_inline,flatten)) auto
freq_pulse(cubic_spline sp, int64_t cycles, bool waittrig, bool sync, bool fb_enable)
{
    assert(cycles >= 4);
    assert((cycles >> 40) == 0);
    auto [isp, shift_len] = convert_pdq_spline_freq(sp.to_array(), cycles);
    auto metadata = raw_param_metadata(shift_len, waittrig, sync, false, fb_enable);
    return pulse(metadata, isp, cycles);
}
static constexpr inline __attribute__((always_inline,flatten)) auto
amp_pulse(cubic_spline sp, int64_t cycles, bool waittrig,
          bool sync=false, bool fb_enable=false)
{
    assert(cycles >= 4);
    assert((cycles >> 40) == 0);
    auto [isp, shift_len] = convert_pdq_spline_amp(sp.to_array(), cycles);
    auto metadata = raw_param_metadata(shift_len, waittrig, sync, false, fb_enable);
    return pulse(metadata, isp, cycles);
}
static constexpr inline __attribute__((always_inline,flatten)) auto
phase_pulse(cubic_spline sp, int64_t cycles, bool waittrig,
            bool sync=false, bool fb_enable=false)
{
    assert(cycles >= 4);
    assert((cycles >> 40) == 0);
    auto [isp, shift_len] = convert_pdq_spline_phase(sp.to_array(), cycles);
    auto metadata = raw_param_metadata(shift_len, waittrig, sync, false, fb_enable);
    return pulse(metadata, isp, cycles);
}

static constexpr inline __attribute__((always_inline,flatten)) uint64_t
raw_frame_metadata(int shift_len, bool waittrig, bool apply_at_end, bool rst_frame,
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
static constexpr inline __attribute__((always_inline,flatten)) auto
frame_pulse(cubic_spline sp, int64_t cycles, bool waittrig, bool apply_at_end,
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

static constexpr inline __attribute__((always_inline,flatten))
JaqalInst stream(JaqalInst pulse)
{
    pulse |= JaqalInst(uint8_t(SeqMode::STREAM)) << Bits::SEQ_MODE;
    pulse |= JaqalInst(1) << Bits::GSEQ_ENABLE;
    return pulse;
}

static constexpr inline __attribute__((always_inline,flatten))
JaqalInst program_PLUT(JaqalInst pulse, uint16_t addr)
{
    assert((addr >> PLUTW) == 0);
    pulse |= JaqalInst(uint8_t(ProgMode::PLUT)) << Bits::PROG_MODE;
    pulse |= JaqalInst(addr) << Bits::PLUT_ADDR;
    return pulse;
}

static constexpr inline __attribute__((always_inline,flatten)) auto
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

static constexpr inline __attribute__((always_inline,flatten)) auto
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

static constexpr inline __attribute__((always_inline,flatten)) auto
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

} // Jaqal_v1_3

extern PyTypeObject &JaqalInst_v1_Type;
extern PyTypeObject &Jaqal_v1_Type;
extern PyTypeObject &JaqalChannelGen_v1_Type;
extern PyTypeObject &JaqalInst_v1_3_Type;
extern PyTypeObject &Jaqal_v1_3_Type;

void init();

}

#endif
