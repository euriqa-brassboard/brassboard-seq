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

#include "rfsoc.h"

namespace brassboard_seq::rfsoc {

namespace {

static inline constexpr bool
test_bits(const JaqalInst &inst, unsigned b1, unsigned b2)
{
    return inst & JaqalInst::mask(b1, b2);
}

struct PDQSpline {
    std::array<int64_t,4> orders;
    int shift;
    double scale;
    cubic_spline get_spline(uint64_t cycles) const
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

        return { forders[0], forders[1], forders[2], forders[3] };
    }
};

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

struct PyJaqalInstBase : PyObject {
    JaqalInst inst;

    template<typename T>
    static auto vectornew(PyObject*, PyObject *const *args, ssize_t nargs,
                          py::tuple kwnames)
    {
        py::check_num_arg(T::ClsName + ".__init__", nargs, 0, 1);
        auto [data] =
            py::parse_pos_or_kw_args<"data">(T::ClsName + ".__init__", args, nargs, kwnames);
        auto self = py::generic_alloc<T>();
        call_constructor(&self->inst);
        if (!data || data.is_none())
            return self;
        auto bytes = py::arg_cast<py::bytes>(data, "data");
        memcpy(&self->inst[0], bytes.data(), std::min((int)bytes.size(), 32));
        return self;
    }

    static PyTypeObject Type;
};
static auto jaqalinstbase_as_number = PyNumberMethods{
    .nb_index = py::unifunc<[] (py::ptr<PyJaqalInstBase> self) {
        return self->inst.to_pylong(); }>
};
PyTypeObject PyJaqalInstBase::Type = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "brassboard_seq.rfsoc_backend.JaqalInstBase",
    .tp_basicsize = sizeof(PyJaqalInstBase),
    .tp_dealloc = py::tp_cxx_dealloc<false,PyJaqalInstBase>,
    .tp_as_number = &jaqalinstbase_as_number,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_richcompare = py::tp_richcompare<[] (py::ptr<PyJaqalInstBase> v1,
                                             py::ptr<> _v2, int op) {
        auto v2 = py::cast<PyJaqalInstBase>(_v2);
        if (!v2 || op != Py_EQ)
            return py::new_not_implemented();
        return py::new_bool(v1.type() == v2.type() && v1->inst == v2->inst);
    }>,
    .tp_methods = (py::meth_table<
                   py::meth_noargs<"to_bytes",[] (py::ptr<PyJaqalInstBase> self) {
                       return self->inst.to_pybytes(); }>>),
};

struct PyJaqalBase {
    static cubic_spline to_spline(py::ptr<> spline)
    {
        auto tu = py::arg_cast<py::tuple>(spline, "spline");
        auto l = tu.size();
        if (l > 4)
            py_throw_format(PyExc_ValueError, "Invalid spline %S", tu);
        auto get = [&] (int i) { return i >= l ? 0.0 : tu.get(i).as_float(); };
        return {get(0), get(1), get(2), get(3)};
    }
    static int to_chn(py::ptr<> _chn)
    {
        int chn = _chn.as_int();
        if (chn < 0 || chn > 7)
            py_throw_format(PyExc_ValueError, "Invalid channel number '%d'", chn);
        return chn;
    }
    static int to_tone(py::ptr<> _tone)
    {
        int tone = _tone.as_int();
        if (tone < 0 || tone > 1)
            py_throw_format(PyExc_ValueError, "Invalid tone number '%d'", tone);
        return tone;
    }
    static int64_t to_cycles(py::ptr<> _cycles)
    {
        auto cycles = py::arg_cast<py::int_>(_cycles, "cycles").as_int<int64_t>();
        if (cycles >> 40)
            py_throw_format(PyExc_ValueError, "Invalid cycle count '%lld'", cycles);
        return cycles;
    }
    template<typename T>
    static auto alloc(const JaqalInst &inst)
    {
        auto self = py::generic_alloc<T>();
        call_constructor(&self->inst, inst);
        return self;
    }
};

} // (anonymous)

namespace Jaqal_v1 {

__attribute__((visibility("protected")))
void ChannelGen::end()
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

namespace {

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
    static auto py_error_msg(Error err)
    {
        switch (err) {
        default:
        case Error::Reserved:
            return "reserved"_py;
        case Error::GLUT_OOB:
            return "glut_oob"_py;
        case Error::SLUT_OOB:
            return "slut_oob"_py;
        case Error::GSEQ_OOB:
            return "gseq_oob"_py;
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
        inst.print(io << "invalid(" << Executor::error_msg(err) << "): ");
    }

    void next()
    {
        io << "\n";
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
        io << " fwd:" << fwd_frame_mask
           << " inv:" << inv_frame_mask;
    }

    py::stringio &io;
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
            io << cspl.order0;
            print_orders(cspl.order1, cspl.order2, cspl.order3,
                         [&] (int, auto order) { io << ", " << order; });
        }
        else {
            io.write_hex(spl.orders[0], spl.orders[0] != 0);
            print_orders(spl.orders[1], spl.orders[2], spl.orders[3],
                         [&] (int i, auto order) {
                             (io << ", ").write_hex(order, order != 0);
                             if (spl.shift) {
                                 io << ">>" << (spl.shift * i);
                             }
                         });
        }
        io << "}";
    }
};

struct DictConverter {
    void invalid(const JaqalInst &inst, Executor::Error err)
    {
        dict.set("type"_py, "invalid"_py);
        dict.set("error"_py, Executor::py_error_msg(err));
        py::stringio io;
        inst.print(io);
        dict.set("inst"_py, io.getvalue());
    }

    void GLUT(uint8_t chn, const uint16_t *gaddrs, const uint16_t *starts,
              const uint16_t *ends, int cnt)
    {
        dict.set("type"_py, "glut"_py);
        dict.set("channel"_py, py::int_cached(chn));
        dict.set("count"_py, py::int_cached(cnt));
        auto py_gaddrs = py::new_list(cnt);
        auto py_starts = py::new_list(cnt);
        auto py_ends = py::new_list(cnt);
        for (int i = 0; i < cnt; i++) {
            py_gaddrs.SET(i, to_py(gaddrs[i]));
            py_starts.SET(i, to_py(starts[i]));
            py_ends.SET(i, to_py(ends[i]));
        }
        dict.set("gaddrs"_py, py_gaddrs);
        dict.set("starts"_py, py_starts);
        dict.set("ends"_py, py_ends);
    }

    void SLUT(uint8_t chn, const uint16_t *saddrs, const uint16_t *paddrs, int cnt)
    {
        dict.set("type"_py, "slut"_py);
        dict.set("channel"_py, py::int_cached(chn));
        dict.set("count"_py, py::int_cached(cnt));
        auto py_saddrs = py::new_list(cnt);
        auto py_paddrs = py::new_list(cnt);
        for (int i = 0; i < cnt; i++) {
            py_saddrs.SET(i, to_py(saddrs[i]));
            py_paddrs.SET(i, to_py(paddrs[i]));
        }
        dict.set("saddrs"_py, py_saddrs);
        dict.set("paddrs"_py, py_paddrs);
    }
    void GSEQ(uint8_t chn, const uint16_t *gaddrs, int cnt, SeqMode m)
    {
        if (m == SeqMode::GATE) {
            dict.set("type"_py, "gseq"_py);
        }
        else if (m == SeqMode::WAIT_ANC) {
            dict.set("type"_py, "wait_anc"_py);
        }
        else if (m == SeqMode::CONT_ANC) {
            dict.set("type"_py, "cont_anc"_py);
        }
        dict.set("channel"_py, py::int_cached(chn));
        dict.set("count"_py, py::int_cached(cnt));
        auto py_gaddrs = py::new_list(cnt);
        for (int i = 0; i < cnt; i++)
            py_gaddrs.SET(i, to_py(gaddrs[i]));
        dict.set("gaddrs"_py, py_gaddrs);
    }

    void param_pulse(int chn, int tone, Executor::ParamType param,
                     const PDQSpline &spl, int64_t cycles, bool waittrig,
                     bool sync, bool enable, bool fb_enable,
                     Executor::PulseTarget tgt)
    {
        pulse_to_dict(tgt, (param == Executor::ParamType::Freq ? "freq"_py :
                            (param == Executor::ParamType::Amp ? "amp"_py :
                             "phase"_py)), chn, tone, cycles, spl);
        dict.set("trig"_py, to_py(waittrig));
        dict.set("sync"_py, to_py(sync));
        dict.set("enable"_py, to_py(enable));
        dict.set("ff"_py, to_py(fb_enable));
    }
    void frame_pulse(int chn, int tone, const PDQSpline &spl, int64_t cycles,
                     bool waittrig, bool apply_eof, bool clr_frame,
                     int fwd_frame_mask, int inv_frame_mask,
                     Executor::PulseTarget tgt)
    {
        pulse_to_dict(tgt, "frame_rot"_py, chn, tone, cycles, spl);
        dict.set("trig"_py, to_py(waittrig));
        dict.set("eof"_py, to_py(apply_eof));
        dict.set("clr"_py, to_py(clr_frame));
        dict.set("fwd"_py, py::int_cached(fwd_frame_mask));
        dict.set("inv"_py, py::int_cached(inv_frame_mask));
    }

    py::dict_ref dict{py::new_dict()};

private:
    void pulse_to_dict(Executor::PulseTarget tgt, PyObject *name, int chn,
                       int tone, int64_t cycles, const PDQSpline &spl)
    {
        if (tgt.type == Executor::PulseTarget::None) {
            dict.set("type"_py, "pulse_data"_py);
        }
        else if (tgt.type == Executor::PulseTarget::PLUT) {
            dict.set("type"_py, "plut"_py);
        }
        else if (tgt.type == Executor::PulseTarget::Stream) {
            dict.set("type"_py, "stream"_py);
        }
        dict.set("channel"_py, py::int_cached(chn));
        if (tgt.type == Executor::PulseTarget::PLUT)
            dict.set("paddr"_py, to_py(tgt.addr));
        dict.set("param"_py, name);
        dict.set("tone"_py, py::int_cached(tone));
        dict.set("cycles"_py, to_py(cycles));
        dict.set("spline_mu"_py, py::new_list(to_py(spl.orders[0]),
                                              to_py(spl.orders[1]),
                                              to_py(spl.orders[2]),
                                              to_py(spl.orders[3])));
        dict.set("spline_shift"_py, py::int_cached(spl.shift));
        auto fspl = spl.get_spline(cycles);
        dict.set("spline"_py, py::new_list(to_py(fspl.order0),
                                           to_py(fspl.order1),
                                           to_py(fspl.order2),
                                           to_py(fspl.order3)));
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

template<typename T> static auto print_inst(T &&v, bool print_float)
{
    py::stringio io;
    Printer printer{io, print_float};
    Executor::execute(printer, std::forward<T>(v));
    return io.getvalue();
}

struct PyInst : PyJaqalInstBase {
    static PyTypeObject Type;
    constexpr static str_literal ClsName = "JaqalInst_v1";
};
PyTypeObject PyInst::Type = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "brassboard_seq.rfsoc_backend.JaqalInst_v1",
    .tp_basicsize = sizeof(PyInst),
    .tp_dealloc = py::tp_cxx_dealloc<false,PyInst>,
    .tp_repr = py::unifunc<[] (py::ptr<PyInst> self) {
        return print_inst(self->inst, false);
    }>,
    .tp_str = py::unifunc<[] (py::ptr<PyInst> self) {
        return print_inst(self->inst, true);
    }>,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_methods = (py::meth_table<
                   py::meth_noargs<"to_dict",[] (py::ptr<PyInst> self) {
                       DictConverter converter;
                       Executor::execute(converter, self->inst);
                       return std::move(converter.dict);
                   }>>),
    .tp_getset = (py::getset_table<
                  py::getset_def<"channel",[] (py::ptr<PyInst> self) {
                      return to_py(get_chn(self->inst)); }>>),
    .tp_base = &PyJaqalInstBase::Type,
    .tp_vectorcall = py::vectorfunc<vectornew<PyInst>>,
};
struct PyJaqal : PyJaqalBase {
    static auto parse_pulse_args(PyObject *const *args, Py_ssize_t nargs)
    {
        int chn = to_chn(py::arg_cast<py::int_>(args[0], "chn"));
        int tone = to_tone(py::arg_cast<py::int_>(args[1], "tone"));
        auto spline = to_spline(args[2]);
        auto cycles = to_cycles(args[3]);
        auto trig = py::arg_cast<py::bool_,true>(args[4], "waittrig").as_bool();
        return std::tuple(chn, tone, spline, cycles, trig);
    }
    static auto parse_param_pulse_args(const char *name, PyObject *const *args,
                                       Py_ssize_t nargs)
    {
        py::check_num_arg(name, nargs, 7, 7);
        auto [chn, tone, spline, cycles, trig] = parse_pulse_args(args, nargs);
        auto sync = py::arg_cast<py::bool_,true>(args[5], "sync").as_bool();
        auto ff = py::arg_cast<py::bool_,true>(args[6], "fb_enable").as_bool();
        return std::tuple(chn, tone, spline, cycles, trig, sync, ff);
    }
    static auto parse_frame_pulse_args(const char *name, PyObject *const *args,
                                       Py_ssize_t nargs)
    {
        py::check_num_arg(name, nargs, 9, 9);
        auto [chn, tone, spline, cycles, trig] = parse_pulse_args(args, nargs);
        auto apply_end = py::arg_cast<py::bool_,true>(args[5], "apply_at_end").as_bool();
        auto rst = py::arg_cast<py::bool_,true>(args[6], "rst_frame").as_bool();
        auto fwd = py::arg_cast<py::int_>(args[7], "fwd_frame_mask").as_int();
        auto inv = py::arg_cast<py::int_>(args[8], "inv_frame_mask").as_int();
        return std::tuple(chn, tone, spline, cycles, trig, apply_end, rst, fwd, inv);
    }
    static PyTypeObject Type;
};
PyTypeObject PyJaqal::Type = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "brassboard_seq.rfsoc_backend.Jaqal_v1",
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_methods = (
        py::meth_table<
        py::meth_fast<"freq_pulse",[] (auto, PyObject *const *args, Py_ssize_t nargs) {
            auto [chn, tone, spline, cycles, trig, sync, ff] =
                parse_param_pulse_args("Jaqal_v1.freq_pulse", args, nargs);
            return alloc<PyInst>(freq_pulse(chn, tone, spline, cycles, trig, sync, ff));
        },"",METH_STATIC>,
        py::meth_fast<"amp_pulse",[] (auto, PyObject *const *args, Py_ssize_t nargs) {
            auto [chn, tone, spline, cycles, trig, sync, ff] =
                parse_param_pulse_args("Jaqal_v1.amp_pulse", args, nargs);
            return alloc<PyInst>(amp_pulse(chn, tone, spline, cycles, trig, sync, ff));
        },"",METH_STATIC>,
        py::meth_fast<"phase_pulse",[] (auto, PyObject *const *args, Py_ssize_t nargs) {
            auto [chn, tone, spline, cycles, trig, sync, ff] =
                parse_param_pulse_args("Jaqal_v1.phase_pulse", args, nargs);
            return alloc<PyInst>(phase_pulse(chn, tone, spline, cycles, trig, sync, ff));
        },"",METH_STATIC>,
        py::meth_fast<"frame_pulse",[] (auto, PyObject *const *args, Py_ssize_t nargs) {
            auto [chn, tone, spline, cycles, trig, apply_end, rst, fwd, inv] =
                parse_frame_pulse_args("Jaqal_v1.frame_pulse", args, nargs);
            return alloc<PyInst>(frame_pulse(chn, tone, spline, cycles, trig,
                                             apply_end, rst, fwd, inv));
        },"",METH_STATIC>,
        py::meth_o<"stream",[] (auto, py::ptr<> _pulse) {
            auto pulse = py::arg_cast<PyInst>(_pulse, "pulse");
            return alloc<PyInst>(stream(pulse->inst));
        },"",METH_STATIC>,
        py::meth_fast<"program_PLUT",[] (auto, PyObject *const *args, Py_ssize_t nargs) {
            py::check_num_arg("Jaqal_v1.program_PLUT", nargs, 2, 2);
            auto pulse = py::arg_cast<PyInst>(args[0], "pulse");
            auto addr = py::arg_cast<py::int_>(args[1], "addr").as_int();
            if (addr < 0 || addr >= 4096)
                py_throw_format(PyExc_ValueError, "Invalid address '%d'", addr);
            return alloc<PyInst>(program_PLUT(pulse->inst, addr));
        },"",METH_STATIC>,
        py::meth_fast<"program_SLUT",[] (auto, PyObject *const *args, Py_ssize_t nargs) {
            py::check_num_arg("Jaqal_v1.program_SLUT", nargs, 3, 3);
            auto chn = to_chn(py::arg_cast<py::int_>(args[0], "chn"));
            py::ptr _saddrs = args[1];
            py::ptr _paddrs = args[2];
            uint16_t saddrs[9];
            uint16_t paddrs[9];
            int n = _saddrs.length();
            if (_paddrs.length() != n)
                py_throw_format(PyExc_ValueError, "Mismatch address length");
            if (n >= 10)
                py_throw_format(PyExc_ValueError, "Too many SLUT addresses to program");
            for (int i = 0; i < n; i++) {
                saddrs[i] = _saddrs.getitem(i).as_int();
                paddrs[i] = _paddrs.getitem(i).as_int();
            }
            return alloc<PyInst>(program_SLUT(chn, saddrs, paddrs, n));
        },"",METH_STATIC>,
        py::meth_fast<"program_GLUT",[] (auto, PyObject *const *args, Py_ssize_t nargs) {
            py::check_num_arg("Jaqal_v1.program_GLUT", nargs, 4, 4);
            auto chn = to_chn(py::arg_cast<py::int_>(args[0], "chn"));
            py::ptr _gaddrs = args[1];
            py::ptr _starts = args[2];
            py::ptr _ends = args[3];
            uint16_t gaddrs[6];
            uint16_t starts[6];
            uint16_t ends[6];
            int n = _gaddrs.length();
            if (_starts.length() != n || _ends.length() != n)
                py_throw_format(PyExc_ValueError, "Mismatch address length");
            if (n >= 7)
                py_throw_format(PyExc_ValueError, "Too many GLUT addresses to program");
            for (int i = 0; i < n; i++) {
                gaddrs[i] = _gaddrs.getitem(i).as_int();
                starts[i] = _starts.getitem(i).as_int();
                ends[i] = _ends.getitem(i).as_int();
            }
            return alloc<PyInst>(program_GLUT(chn, gaddrs, starts, ends, n));
        },"",METH_STATIC>,
        py::meth_fast<"sequence",[] (auto, PyObject *const *args, Py_ssize_t nargs) {
            py::check_num_arg("Jaqal_v1.sequence", nargs, 3, 3);
            auto chn = to_chn(py::arg_cast<py::int_>(args[0], "chn"));
            auto mode = py::arg_cast<py::int_>(args[1], "mode").as_int();
            py::ptr _gaddrs = args[2];
            uint16_t gaddrs[24];
            int n = _gaddrs.length();
            if (n >= 25)
                py_throw_format(PyExc_ValueError, "Too many GLUT addresses to sequence");
            for (int i = 0; i < n; i++)
                gaddrs[i] = _gaddrs.getitem(i).as_int();
            if (mode != int(SeqMode::GATE) && mode != int(SeqMode::WAIT_ANC) &&
                mode != int(SeqMode::CONT_ANC))
                py_throw_format(PyExc_ValueError, "Invalid sequencing mode %d.", mode);
            return alloc<PyInst>(sequence(chn, (SeqMode)mode, gaddrs, n));
        },"",METH_STATIC>,
        py::meth_fastkw<"dump_insts",[] (auto, PyObject *const *args, Py_ssize_t nargs,
                                         py::tuple kwnames) {
            py::check_num_arg("Jaqal_v1.dump_insts", nargs, 1, 2);
            auto b = py::arg_cast<py::bytes>(args[0], "b");
            auto [_pfloat] =
                py::parse_pos_or_kw_args<"print_float">("Jaqal_v1.dump_insts",
                                                        args + 1, nargs - 1, kwnames);
            bool pfloat = true;
            if (_pfloat)
                pfloat = py::arg_cast<py::bool_,true>(_pfloat, "print_float").as_bool();
            return print_inst(std::span(b.data(), b.size()), pfloat);
        },"",METH_STATIC>,
        py::meth_o<"extract_pulses",[] (auto, py::ptr<> _b) {
            auto b = py::arg_cast<py::bytes>(_b, "b");
            PulseSequencer sequencer;
            Executor::execute(sequencer, std::span(b.data(), b.size()));
            return py::new_nlist(sequencer.pulses.size(), [&] (int i) {
                return alloc<PyInst>(sequencer.pulses[i]);
            });
        },"",METH_STATIC>>),
};
struct PyChannelGen : PyObject {
    ChannelGen chn_gen;

    static PyTypeObject Type;
};
PyTypeObject PyChannelGen::Type = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "brassboard_seq.rfsoc_backend.JaqalChannelGen_v1",
    .tp_basicsize = sizeof(PyChannelGen),
    .tp_dealloc = py::tp_cxx_dealloc<false,PyChannelGen>,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_methods = (
        py::meth_table<
        py::meth_fast<"add_pulse",[] (py::ptr<PyChannelGen> self, PyObject *const *args,
                                      Py_ssize_t nargs) {
            py::check_num_arg("JaqalChannelGen_v1.add_pulse", nargs, 2, 2);
            auto pulse = py::arg_cast<PyInst>(args[0], "pulse");
            auto cycle = py::arg_cast<py::int_>(args[1], "cycle").as_int<int64_t>();
            self->chn_gen.add_pulse(pulse->inst, cycle);
        }>,
        py::meth_noargs<"clear",[] (py::ptr<PyChannelGen> self) {
            self->chn_gen.clear();
        }>,
        py::meth_noargs<"end",[] (py::ptr<PyChannelGen> self) {
            self->chn_gen.end();
        }>,
        py::meth_noargs<"get_plut",[] (py::ptr<PyChannelGen> self) {
            auto &pulses = self->chn_gen.pulses.pulses;
            auto res = py::new_list(pulses.size());
            for (auto [inst, i]: pulses)
                res.SET(i, PyJaqal::alloc<PyInst>(inst));
            return res;
        }>,
        py::meth_noargs<"get_slut",[] (py::ptr<PyChannelGen> self) {
            return to_py(self->chn_gen.slut);
        }>,
        py::meth_noargs<"get_glut",[] (py::ptr<PyChannelGen> self) {
            return to_py(self->chn_gen.glut);
        }>,
        py::meth_noargs<"get_gseq",[] (py::ptr<PyChannelGen> self) {
            auto &gate_ids = self->chn_gen.gate_ids;
            return py::new_nlist(gate_ids.size(), [&] (int i) {
                auto gate = gate_ids[i];
                return py::new_tuple(to_py(gate.time), to_py(gate.id));
            });
        }>>),
    .tp_vectorcall = py::vectorfunc<[] (auto, PyObject *const*,
                                        ssize_t nargs, py::tuple kwnames) {
        py::check_num_arg("JaqalChannelGen_v1.__init__", nargs, 0, 0);
        py::check_no_kwnames("JaqalChannelGen_v1.__init__", kwnames);
        auto self = py::generic_alloc<PyChannelGen>();
        call_constructor(&self->chn_gen);
        return self;
    }>,
};

} // (anonymous)
} // Jaqal_v1

namespace Jaqal_v1_3 {

namespace {

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

struct GateConstructor {
    GateConstructor(ChannelGen &gen)
        : gen(gen)
    {
    }
    ChannelGen &gen;
    int nblocks;
    std::vector<ChannelGen::Block*> linear_blocks;
    std::map<std::pair<int8_t,int>,int> single_gates;

    std::vector<int> substrs_cache;
    void sort_substr(SubStrInfoSA sa, int nrep)
    {
        substrs_cache.clear();
        auto sa_begin = sa.sa_begin;
        auto sa_end = sa_begin + nrep;
        for (int sa_idx = sa_begin; sa_idx < sa_end; sa_idx++)
            substrs_cache.push_back(pulse_sa[sa_idx]);
        std::ranges::sort(substrs_cache);
    }

    std::vector<int> pulse_str;
    std::vector<int> pulse_sa;
    std::vector<int> pulse_rk;
    std::vector<int> pulse_height;
    // Map of start index -> (length, gate_id)
    std::map<int,std::pair<int,int>> substr_map;
    std::map<std::pair<int,int>,SubStrInfos> substrs;
    void discover_gate()
    {
        // Convert all pulse ids to a linear sequence so that we can use suffix array.
        assert(pulse_str.empty());
        // pulse_str value range:
        // 0: end
        // 1-nblocks: block end
        // >=nblocks + 1: normal pulses (pulse_id << 3 | param)

        for (auto *block: linear_blocks) {
            auto block_id = block->block_id;
            for (auto &pulse: block->pulse_ids)
                pulse_str.push_back(((pulse.pulse_id << 3) | pulse.param) + nblocks + 1);
            pulse_str.push_back(block_id + 1);
        }
        int npulses = pulse_str.size();
        pulse_str.push_back(0);
        get_suffix_array(pulse_sa, pulse_str, pulse_rk);
        order_to_rank(pulse_rk, pulse_sa);
        get_height_array(pulse_height, pulse_str, pulse_sa, pulse_rk);
        pulse_rk.clear();
        // Sort substrings according to repetition and length
        foreach_max_range(pulse_height, [&] (int i0, int i1, int str_len) {
            if (str_len <= 1)
                return;
            auto sa_begin = i0 + 1;
            auto nrep = i1 - i0 + 2;
            substrs[{nrep, str_len}].sas.push_back({ sa_begin });
        });
        pulse_height.clear();

        auto new_gate = [&] {
            int gate_id = gen.gates.size();
            return std::pair(&gen.gates.emplace_back(), gate_id);
        };
        auto add_gate = [&] (int start, int len, int nrep) {
            auto [gate, gate_id] = new_gate();
            assert(len > 1);
            gate->nrep = nrep;
            for (int i = 0; i < len; i++) {
                auto pulse_id = pulse_str[start + i] - (nblocks + 1);
                gate->pulse_ids.push_back({pulse_id & 7, pulse_id >> 3});
            }
            return gate_id;
        };

        // Identify non-overlapping repeating pulse sequences.

        auto check_substr = [&] (std::vector<int> &substr_idxs, int str_len) {
            // Note that `substr_idxs` may alias `substrs_cache`.
            // but it doesn't alias anything else that's still in use elsewhere
            // so it can be freely mutated.

            assert(str_len > 1);
            int max_len = str_len;
            int nsubstrs = (int)substr_idxs.size();
            int last_substr = npulses;
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
            auto gate_id = add_gate(substr_idxs[0], max_len, new_nsubstrs);
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
            for (auto &vec: it->second.vecs)
                check_substr(vec.strs, str_len);
            substrs.erase(it);
        }
        pulse_sa.clear();

        // Divide pulses into gates
        int cur_block_id = 0;
        auto cur_block = linear_blocks[0];
        int block_pulse_offset = 0;
        auto sequence_gate = [&] (int gate_id, int pulse_pos) {
            cur_block->gates.push_back({pulse_pos - block_pulse_offset, gate_id});
        };
        auto next_block = [&] (int pulse_idx_start) {
            cur_block_id += 1;
            if (cur_block_id >= nblocks)
                return;
            cur_block = linear_blocks[cur_block_id];
            block_pulse_offset = pulse_idx_start;
        };
        auto process_range = [&] (int start, int end) {
            int gate_id = 0;
            ChannelGen::Gate *cur_gate = nullptr;
            std::pair<int8_t,int> first_pulse = {-1, 0};
            auto finish_gate = [&] {
                if (cur_gate) {
                    sequence_gate(gate_id, start);
                    cur_gate = nullptr;
                    assert(first_pulse.first < 0);
                }
                else if (first_pulse.first >= 0) {
                    auto [it, inserted] = single_gates.insert({first_pulse, 0});
                    if (inserted) {
                        auto [gate, gate_id] = new_gate();
                        gate->nrep = 1;
                        gate->pulse_ids.push_back(first_pulse);
                        it->second = gate_id;
                        sequence_gate(gate_id, start);
                    }
                    else {
                        gen.gates[it->second].nrep += 1;
                        sequence_gate(it->second, start);
                    }
                    first_pulse.first = -1;
                }
            };
            auto push_pulse = [&] (int pulse_id) {
                std::pair<int8_t,int> new_pulse = {pulse_id & 7, pulse_id >> 3};
                if (first_pulse.first >= 0) {
                    std::tie(cur_gate, gate_id) = new_gate();
                    cur_gate->nrep = 1;
                    cur_gate->pulse_ids.push_back(first_pulse);
                    first_pulse.first = -1;
                }
                else if (!cur_gate) {
                    first_pulse = new_pulse;
                    return;
                }
                cur_gate->pulse_ids.push_back(new_pulse);
            };
            for (auto pulse_idx = start; pulse_idx < end; pulse_idx++) {
                auto p = pulse_str[pulse_idx];
                if (p <= nblocks) {
                    finish_gate();
                    start = pulse_idx + 1;
                    next_block(start);
                    continue;
                }
                push_pulse(p - (nblocks + 1));
            }
            finish_gate();
        };

        int next_pulse = 0;
        for (auto [str_idx, substr_info]: substr_map) {
            assert(str_idx >= next_pulse);
            if (str_idx > next_pulse)
                process_range(next_pulse, str_idx);
            auto [max_len, gate_id] = substr_info;
            sequence_gate(gate_id, str_idx);
            next_pulse = str_idx + max_len;
        }
        if (npulses > next_pulse)
            process_range(next_pulse, npulses);

        substr_map.clear();
        pulse_str.clear();
    }
    void even_gate_length(ChannelGen::SyncBlock &sblock)
    {
        int max_gate_len = 0;
        int min_gate_len = INT_MAX;
        for (auto &block: sblock.blocks) {
            int gate_len = block.gates.size();
            min_gate_len = std::min(min_gate_len, gate_len);
            max_gate_len = std::max(max_gate_len, gate_len);
        }
        if (max_gate_len == min_gate_len)
            return;
        for (auto &block: sblock.blocks) {
            int gate_len = block.gates.size();
            if (max_gate_len == gate_len)
                continue;
            // First break up subsequences that are only used once
            for (int i = gate_len - 1; i >= 0 && max_gate_len > gate_len; i--) {
                auto [pulse_idx, gate_id] = block.gates[i];
                if (gen.gates[gate_id].nrep != 1)
                    continue;
                int gate_sz = gen.gates[gate_id].pulse_ids.size();
                if (gate_sz <= 1)
                    continue;
                auto peel_off = std::min(gate_sz - 1, max_gate_len - gate_len);
                int new_gate_start = gen.gates.size();
                gen.gates.resize(new_gate_start + peel_off);
                block.gates.resize(gate_len + peel_off);
                memmove(&block.gates[i + 1 + peel_off], &block.gates[i + 1],
                        (gate_len - i - 1) * sizeof(block.gates[0]));
                int first_gate_sz = gate_sz - peel_off;
                auto &orig_gate_pulses = gen.gates[gate_id].pulse_ids;
                for (int j = 0; j < peel_off; j++) {
                    block.gates[i + 1 + j] = { pulse_idx + first_gate_sz + j,
                        new_gate_start + j};
                    auto &gate = gen.gates[new_gate_start + j];
                    gate.nrep = 1;
                    gate.pulse_ids.push_back(orig_gate_pulses[first_gate_sz + j]);
                }
                orig_gate_pulses.resize(first_gate_sz);
                gate_len = gate_len + peel_off;
            }
            assert(max_gate_len >= gate_len);
            if (max_gate_len > gate_len)
                continue;
            // Maybe we could split up long constant pulses in the future
            // Fill with dummy gates for now.
            block.gates.resize(max_gate_len, { -1, -1 });
        }
    }
    void process_seq()
    {
        int nblocks = 0;
        for (auto &sblock: gen.sblocks) {
            auto n = sblock.blocks.size();
            for (int i = 0; i < n; i++) {
                auto &block = sblock.blocks[i];
                block.block_id = nblocks + i;
                linear_blocks.push_back(&block);
                std::ranges::sort(block.pulse_ids, [] (auto &a, auto &b) {
                    // Move long pulses and phase pulses to the front
                    // They are more likely to be different between otherwise similar
                    // gate sequence.
                    return (std::tuple(a.time, -a.len, -a.param) <
                            std::tuple(b.time, -b.len, -b.param));
                });
            }
            nblocks += n;
        }
        assert(nblocks > 0);
        this->nblocks = nblocks;

        discover_gate();
        for (auto &sblock: gen.sblocks)
            even_gate_length(sblock);
    }
};

}

__attribute__((flatten))
void ChannelGen::construct_gates()
{
    GateConstructor constructor(*this);
    constructor.process_seq();
}

namespace {

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
    static auto py_error_msg(Error err)
    {
        switch (err) {
        default:
        case Error::Reserved:
            return "reserved"_py;
        case Error::GLUT_OOB:
            return "glut_oob"_py;
        case Error::SLUT_OOB:
            return "slut_oob"_py;
        case Error::GSEQ_OOB:
            return "gseq_oob"_py;
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
    static cubic_spline freq_spline(PDQSpline spl, uint64_t cycles)
    {
        spl.scale = output_clock / double(1ll << 40);
        return spl.get_spline(cycles);
    }
    static cubic_spline amp_spline(PDQSpline spl, uint64_t cycles)
    {
        spl.scale = 1 / double(((1ll << 16) - 1ll) << 23);
        return spl.get_spline(cycles);
    }
    static cubic_spline phase_spline(PDQSpline spl, uint64_t cycles)
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
        inst.print(io << "invalid(" << Executor::error_msg(err) << "): ");
    }

    void next()
    {
        io << "\n";
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
                io << cspl.order0;
                print_orders(cspl.order1, cspl.order2, cspl.order3,
                             [&] (int, auto order) { io << ", " << order; });
                io << "}";
            };
            print_spl("freq", Executor::freq_spline, freq);
            print_spl("amp", Executor::amp_spline, amp);
            print_spl("phase", Executor::phase_spline, phase);
        }
        else {
            io << " {";
            io.write_hex(spl.orders[0], spl.orders[0] != 0);
            print_orders(spl.orders[1], spl.orders[2], spl.orders[3],
                         [&] (int i, auto order) {
                             (io << ", ").write_hex(order, order != 0);
                             if (spl.shift) {
                                 io << ">>" << (spl.shift * i);
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
            io << " fwd:" << int(meta.fwd_frame_mask)
               << " inv:" << int(meta.inv_frame_mask);
        }
    }

    py::stringio &io;
    bool print_float{true};
private:
    void print_list_ele(const char *name, bool cond, IsFirst &first)
    {
        if (!cond)
            return;
        if (!first.get())
            io << ",";
        io << name;
    }
    void print_mod_type(ModTypeMask mod_type, bool force_brace)
    {
        bool use_brace = force_brace || (std::popcount(uint8_t(mod_type)) != 1);
        IsFirst first;
        if (use_brace)
            io << "{";
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
            IsFirst first;
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
        dict.set("type"_py, "invalid"_py);
        dict.set("error"_py, Executor::py_error_msg(err));
        py::stringio io;
        inst.print(io);
        dict.set("inst"_py, io.getvalue());
    }

    void GLUT(uint8_t chn_mask, const uint16_t *gaddrs, const uint16_t *starts,
              const uint16_t *ends, int cnt)
    {
        dict.set("type"_py, "glut"_py);
        set_channels(chn_mask);
        dict.set("count"_py, py::int_cached(cnt));
        auto py_gaddrs = py::new_list(cnt);
        auto py_starts = py::new_list(cnt);
        auto py_ends = py::new_list(cnt);
        for (int i = 0; i < cnt; i++) {
            py_gaddrs.SET(i, to_py(gaddrs[i]));
            py_starts.SET(i, to_py(starts[i]));
            py_ends.SET(i, to_py(ends[i]));
        }
        dict.set("gaddrs"_py, py_gaddrs);
        dict.set("starts"_py, py_starts);
        dict.set("ends"_py, py_ends);
    }

    void SLUT(uint8_t chn_mask, const uint16_t *saddrs,
              const ModTypeMask *mod_types, const uint16_t *paddrs, int cnt)
    {
        dict.set("type"_py, "slut"_py);
        set_channels(chn_mask);
        dict.set("count"_py, py::int_cached(cnt));
        auto py_saddrs = py::new_list(cnt);
        auto py_modtypes = py::new_list(cnt);
        auto py_paddrs = py::new_list(cnt);
        for (int i = 0; i < cnt; i++) {
            py_saddrs.SET(i, to_py(saddrs[i]));
            py_paddrs.SET(i, to_py(paddrs[i]));
            py_modtypes.SET(i, mod_type_list(mod_types[i]));
        }
        dict.set("saddrs"_py, py_saddrs);
        dict.set("modtypes"_py, py_modtypes);
        dict.set("paddrs"_py, py_paddrs);
    }
    void GSEQ(uint8_t chn_mask, const uint16_t *gaddrs, int cnt, SeqMode m)
    {
        if (m == SeqMode::GATE) {
            dict.set("type"_py, "gseq"_py);
        }
        else if (m == SeqMode::WAIT_ANC) {
            dict.set("type"_py, "wait_anc"_py);
        }
        else if (m == SeqMode::CONT_ANC) {
            dict.set("type"_py, "cont_anc"_py);
        }
        set_channels(chn_mask);
        dict.set("count"_py, py::int_cached(cnt));
        auto py_gaddrs = py::new_list(cnt);
        for (int i = 0; i < cnt; i++)
            py_gaddrs.SET(i, to_py(gaddrs[i]));
        dict.set("gaddrs"_py, py_gaddrs);
    }

    void pulse(int chn_mask, ModTypeMask mod_type, const PDQSpline &spl,
               int64_t cycles, Executor::PulseMeta meta, Executor::PulseTarget tgt)
    {
        if (tgt.type == Executor::PulseTarget::None) {
            dict.set("type"_py, "pulse_data"_py);
        }
        else if (tgt.type == Executor::PulseTarget::PLUT) {
            dict.set("type"_py, "plut"_py);
        }
        else if (tgt.type == Executor::PulseTarget::Stream) {
            dict.set("type"_py, "stream"_py);
        }
        bool is_plut = tgt.type == Executor::PulseTarget::PLUT;
        if (is_plut) {
            dict.set("paddr"_py, to_py(tgt.addr));
        }
        else {
            dict.set("modtype"_py, mod_type_list(mod_type));
        }
        set_channels(chn_mask);
        dict.set("cycles"_py, to_py(cycles));
        dict.set("spline_mu"_py, py::new_list(to_py(spl.orders[0]), to_py(spl.orders[1]),
                                              to_py(spl.orders[2]), to_py(spl.orders[3])));
        dict.set("spline_shift"_py, py::int_cached(spl.shift));
        bool freq = mod_type & (FRQMOD0_MASK | FRQMOD1_MASK) || is_plut;
        bool amp = mod_type & (AMPMOD0_MASK | AMPMOD1_MASK) || is_plut;
        bool phase = mod_type & (PHSMOD0_MASK | PHSMOD1_MASK |
                                 FRMROT0_MASK | FRMROT1_MASK) || is_plut;
        bool unprefix_spline = (int(freq) + int(amp) + int(phase)) == 1;
        auto set_spline = [&] (PyObject *name, auto &&cb, bool cond) {
            if (!cond)
                return;
            auto fspl = cb(spl, cycles);
            auto py_fspl = py::new_list(to_py(fspl.order0), to_py(fspl.order1),
                                        to_py(fspl.order2), to_py(fspl.order3));
            if (unprefix_spline)
                dict.set("spline"_py, py_fspl);
            dict.set(name, py_fspl);
        };
        set_spline("spline_freq"_py, Executor::freq_spline, freq);
        set_spline("spline_amp"_py, Executor::amp_spline, amp);
        set_spline("spline_phase"_py, Executor::phase_spline, phase);
        dict.set("trig"_py, to_py(meta.trig));
        dict.set("sync"_py, to_py(meta.sync));
        dict.set("enable"_py, to_py(meta.en));
        dict.set("ff"_py, to_py(meta.fb));
        dict.set("eof"_py, to_py(meta.apply_eof));
        dict.set("clr"_py, to_py(meta.clr_frame));
        dict.set("fwd"_py, py::int_cached(meta.fwd_frame_mask));
        dict.set("inv"_py, py::int_cached(meta.inv_frame_mask));
    }

    py::dict_ref dict{py::new_dict()};

private:
    void set_channels(uint8_t chn_mask)
    {
        int nchns = std::popcount(chn_mask);
        int chn_added = 0;
        auto chns = py::new_list(nchns);
        for (int chn = 0; chn < 8; chn++) {
            if (!((chn_mask >> chn) & 1))
                continue;
            if (nchns == 1)
                dict.set("channel"_py, py::int_cached(chn));
            chns.SET(chn_added, to_py(chn));
            chn_added++;
        }
        assert(nchns == chn_added);
        dict.set("channels"_py, chns);
    }
    py::ref<> mod_type_list(ModTypeMask mod_type)
    {
        int nmasks = std::popcount(uint8_t(mod_type));
        auto names = py::new_list(nmasks);
        int name_added = 0;
        auto add_name = [&] (py::ptr<> name, ModTypeMask mask) {
            if (!(mod_type & mask))
                return;
            names.SET(name_added, name);
            name_added++;
        };
        add_name("freq0"_py, FRQMOD0_MASK);
        add_name("amp0"_py, AMPMOD0_MASK);
        add_name("phase0"_py, PHSMOD0_MASK);
        add_name("freq1"_py, FRQMOD1_MASK);
        add_name("amp1"_py, AMPMOD1_MASK);
        add_name("phase1"_py, PHSMOD1_MASK);
        add_name("frame_rot0"_py, FRMROT0_MASK);
        add_name("frame_rot1"_py, FRMROT1_MASK);
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

template<typename T> static auto print_inst(T &&v, bool print_float)
{
    py::stringio io;
    Printer printer{io, print_float};
    Executor::execute(printer, std::forward<T>(v));
    return io.getvalue();
}

struct PyInst : PyJaqalInstBase {
    static PyTypeObject Type;
    constexpr static str_literal ClsName = "JaqalInst_v1_3";
};
PyTypeObject PyInst::Type = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "brassboard_seq.rfsoc_backend.JaqalInst_v1_3",
    .tp_basicsize = sizeof(PyInst),
    .tp_dealloc = py::tp_cxx_dealloc<false,PyInst>,
    .tp_repr = py::unifunc<[] (py::ptr<PyInst> self) {
        return print_inst(self->inst, false);
    }>,
    .tp_str = py::unifunc<[] (py::ptr<PyInst> self) {
        return print_inst(self->inst, true);
    }>,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_methods = (py::meth_table<
                   py::meth_noargs<"to_dict",[] (py::ptr<PyInst> self) {
                       DictConverter converter;
                       Executor::execute(converter, self->inst);
                       return std::move(converter.dict);
                   }>>),
    .tp_getset = (py::getset_table<
                  py::getset_def<"channel_mask",[] (py::ptr<PyInst> self) {
                      return to_py(get_chn_mask(self->inst)); }>,
                  py::getset_def<"channels",[] (py::ptr<PyInst> self) {
                      auto res = py::new_list(0);
                      auto chn_mask = get_chn_mask(self->inst);
                      for (int i = 0; i < 8; i++) {
                          if ((chn_mask >> i) & 1) {
                              res.append(to_py(i));
                          }
                      }
                      return res;
                  }>>),
    .tp_base = &PyJaqalInstBase::Type,
    .tp_vectorcall = py::vectorfunc<vectornew<PyInst>>,
};
struct PyJaqal : PyJaqalBase {
    static ModTypeMask get_modtype_mask(py::ptr<> modtype)
    {
        static py::dict modtype_name_map = [] {
            auto map = py::new_dict();
            map.set("freq0"_py, py::int_cached(int(FRQMOD0_MASK)));
            map.set("amp0"_py, py::int_cached(int(AMPMOD0_MASK)));
            map.set("phase0"_py, py::int_cached(int(PHSMOD0_MASK)));
            map.set("frame_rot0"_py, py::int_cached(int(FRMROT0_MASK)));
            map.set("freq1"_py, py::int_cached(int(FRQMOD1_MASK)));
            map.set("amp1"_py, py::int_cached(int(AMPMOD1_MASK)));
            map.set("phase1"_py, py::int_cached(int(PHSMOD1_MASK)));
            map.set("frame_rot1"_py, py::int_cached(int(FRMROT1_MASK)));
            return map.rel();
        } ();
        if (auto _modmask = py::cast<py::int_>(modtype)) {
            auto modmask = _modmask.as_int();
            if (modmask < 0 || modmask > 255)
                py_throw_format(PyExc_ValueError, "Invalid mod type '%d'", modmask);
            return (ModTypeMask)modmask;
        }
        else if (auto modname = py::cast<py::str>(modtype)) {
            auto mask = modtype_name_map.try_get(modname);
            if (!mask)
                py_throw_format(PyExc_ValueError, "Invalid mod type '%U'", modtype);
            return (ModTypeMask)mask.as_int();
        }
        int modmask = 0;
        for (auto name: modtype.generic_iter()) {
            auto mask = modtype_name_map.try_get(name);
            if (!mask)
                py_throw_format(PyExc_ValueError, "Invalid mod type '%S'", name);
            modmask |= mask.as_int();
        }
        return (ModTypeMask)modmask;
    }
    static int get_chn_mask(py::ptr<> channels)
    {
        if (auto chn = py::cast<py::int_>(channels))
            return 1 << to_chn(chn);
        int chnmask = 0;
        for (auto chn: channels.generic_iter())
            chnmask |= 1 << to_chn(py::arg_cast<py::int_>(chn, "chn"));
        return chnmask;
    }

    static auto parse_pulse_args(PyObject *const *args, Py_ssize_t nargs)
    {
        auto spline = to_spline(args[0]);
        auto cycles = to_cycles(args[1]);
        auto trig = py::arg_cast<py::bool_,true>(args[2], "waittrig").as_bool();
        return std::tuple(spline, cycles, trig);
    }
    static auto parse_param_pulse_args(const char *name, PyObject *const *args,
                                       Py_ssize_t nargs)
    {
        py::check_num_arg(name, nargs, 5, 5);
        auto [spline, cycles, trig] = parse_pulse_args(args, nargs);
        auto sync = py::arg_cast<py::bool_,true>(args[3], "sync").as_bool();
        auto ff = py::arg_cast<py::bool_,true>(args[4], "fb_enable").as_bool();
        return std::tuple(spline, cycles, trig, sync, ff);
    }
    static auto parse_frame_pulse_args(const char *name, PyObject *const *args,
                                       Py_ssize_t nargs)
    {
        py::check_num_arg(name, nargs, 7, 7);
        auto [spline, cycles, trig] = parse_pulse_args(args, nargs);
        auto apply_end = py::arg_cast<py::bool_,true>(args[3], "apply_at_end").as_bool();
        auto rst = py::arg_cast<py::bool_,true>(args[4], "rst_frame").as_bool();
        auto fwd = py::arg_cast<py::int_>(args[5], "fwd_frame_mask").as_int();
        auto inv = py::arg_cast<py::int_>(args[6], "inv_frame_mask").as_int();
        return std::tuple(spline, cycles, trig, apply_end, rst, fwd, inv);
    }
    static PyTypeObject Type;
};
PyTypeObject PyJaqal::Type = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "brassboard_seq.rfsoc_backend.Jaqal_v1_3",
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_methods = (
        py::meth_table<
        py::meth_fast<"freq_pulse",[] (auto, PyObject *const *args, Py_ssize_t nargs) {
            auto [spline, cycles, trig, sync, ff] =
                parse_param_pulse_args("Jaqal_v1_3.freq_pulse", args, nargs);
            return alloc<PyInst>(freq_pulse(spline, cycles, trig, sync, ff));
        },"",METH_STATIC>,
        py::meth_fast<"amp_pulse",[] (auto, PyObject *const *args, Py_ssize_t nargs) {
            auto [spline, cycles, trig, sync, ff] =
                parse_param_pulse_args("Jaqal_v1_3.amp_pulse", args, nargs);
            return alloc<PyInst>(amp_pulse(spline, cycles, trig, sync, ff));
        },"",METH_STATIC>,
        py::meth_fast<"phase_pulse",[] (auto, PyObject *const *args, Py_ssize_t nargs) {
            auto [spline, cycles, trig, sync, ff] =
                parse_param_pulse_args("Jaqal_v1_3.phase_pulse", args, nargs);
            return alloc<PyInst>(phase_pulse(spline, cycles, trig, sync, ff));
        },"",METH_STATIC>,
        py::meth_fast<"frame_pulse",[] (auto, PyObject *const *args, Py_ssize_t nargs) {
            auto [spline, cycles, trig, apply_end, rst, fwd, inv] =
                parse_frame_pulse_args("Jaqal_v1_3.frame_pulse", args, nargs);
            return alloc<PyInst>(frame_pulse(spline, cycles, trig,
                                             apply_end, rst, fwd, inv));
        },"",METH_STATIC>,
        py::meth_fast<"apply_channel_mask",[] (auto, PyObject *const *args, Py_ssize_t nargs) {
            py::check_num_arg("Jaqal_v1_3.apply_channel_mask", nargs, 2, 2);
            auto pulse = py::arg_cast<PyInst>(args[0], "pulse");
            auto chnmask = get_chn_mask(args[1]);
            return alloc<PyInst>(apply_channel_mask(pulse->inst, chnmask));
        },"",METH_STATIC>,
        py::meth_fast<"apply_modtype_mask",[] (auto, PyObject *const *args, Py_ssize_t nargs) {
            py::check_num_arg("Jaqal_v1_3.apply_modtype_mask", nargs, 2, 2);
            auto pulse = py::arg_cast<PyInst>(args[0], "pulse");
            auto modmask = get_modtype_mask(args[1]);
            return alloc<PyInst>(apply_modtype_mask(pulse->inst, modmask));
        },"",METH_STATIC>,
        py::meth_o<"stream",[] (auto, py::ptr<> _pulse) {
            auto pulse = py::arg_cast<PyInst>(_pulse, "pulse");
            return alloc<PyInst>(stream(pulse->inst));
        },"",METH_STATIC>,
        py::meth_fast<"program_PLUT",[] (auto, PyObject *const *args, Py_ssize_t nargs) {
            py::check_num_arg("Jaqal_v1_3.program_PLUT", nargs, 2, 2);
            auto pulse = py::arg_cast<PyInst>(args[0], "pulse");
            auto addr = py::arg_cast<py::int_>(args[1], "addr").as_int();
            if (addr < 0 || addr >= 4096)
                py_throw_format(PyExc_ValueError, "Invalid address '%d'", addr);
            return alloc<PyInst>(program_PLUT(pulse->inst, addr));
        },"",METH_STATIC>,
        py::meth_fast<"program_SLUT",[] (auto, PyObject *const *args, Py_ssize_t nargs) {
            py::check_num_arg("Jaqal_v1_3.program_SLUT", nargs, 4, 4);
            auto chnmask = get_chn_mask(args[0]);
            py::ptr _saddrs = args[1];
            py::ptr _paddrs = args[2];
            py::ptr _modtypes = args[3];
            uint16_t saddrs[6];
            uint16_t paddrs[6];
            ModTypeMask modtypes[6];
            int n = _saddrs.length();
            if (_paddrs.length() != n || _modtypes.length() != n)
                py_throw_format(PyExc_ValueError, "Mismatch address length");
            if (n >= 7)
                py_throw_format(PyExc_ValueError, "Too many SLUT addresses to program");
            for (int i = 0; i < n; i++) {
                saddrs[i] = _saddrs.getitem(i).as_int();
                paddrs[i] = _paddrs.getitem(i).as_int();
                modtypes[i] = get_modtype_mask(_modtypes.getitem(i));
            }
            return alloc<PyInst>(program_SLUT(chnmask, saddrs, modtypes, paddrs, n));
        },"",METH_STATIC>,
        py::meth_fast<"program_GLUT",[] (auto, PyObject *const *args, Py_ssize_t nargs) {
            py::check_num_arg("Jaqal_v1_3.program_GLUT", nargs, 4, 4);
            auto chnmask = get_chn_mask(args[0]);
            py::ptr _gaddrs = args[1];
            py::ptr _starts = args[2];
            py::ptr _ends = args[3];
            uint16_t gaddrs[6];
            uint16_t starts[6];
            uint16_t ends[6];
            int n = _gaddrs.length();
            if (_starts.length() != n || _ends.length() != n)
                py_throw_format(PyExc_ValueError, "Mismatch address length");
            if (n >= 7)
                py_throw_format(PyExc_ValueError, "Too many GLUT addresses to program");
            for (int i = 0; i < n; i++) {
                gaddrs[i] = _gaddrs.getitem(i).as_int();
                starts[i] = _starts.getitem(i).as_int();
                ends[i] = _ends.getitem(i).as_int();
            }
            return alloc<PyInst>(program_GLUT(chnmask, gaddrs, starts, ends, n));
        },"",METH_STATIC>,
        py::meth_fast<"sequence",[] (auto, PyObject *const *args, Py_ssize_t nargs) {
            py::check_num_arg("Jaqal_v1_3.sequence", nargs, 3, 3);
            auto chnmask = get_chn_mask(args[0]);
            auto mode = py::arg_cast<py::int_>(args[1], "mode").as_int();
            py::ptr _gaddrs = args[2];
            uint16_t gaddrs[20];
            int n = _gaddrs.length();
            if (n >= 21)
                py_throw_format(PyExc_ValueError, "Too many GLUT addresses to sequence");
            for (int i = 0; i < n; i++)
                gaddrs[i] = _gaddrs.getitem(i).as_int();
            if (mode != int(SeqMode::GATE) && mode != int(SeqMode::WAIT_ANC) &&
                mode != int(SeqMode::CONT_ANC))
                py_throw_format(PyExc_ValueError, "Invalid sequencing mode %d.", mode);
            return alloc<PyInst>(sequence(chnmask, (SeqMode)mode, gaddrs, n));
        },"",METH_STATIC>,
        py::meth_fastkw<"dump_insts",[] (auto, PyObject *const *args, Py_ssize_t nargs,
                                         py::tuple kwnames) {
            py::check_num_arg("Jaqal_v1_3.dump_insts", nargs, 1, 2);
            auto b = py::arg_cast<py::bytes>(args[0], "b");
            auto [_pfloat] =
                py::parse_pos_or_kw_args<"print_float">("Jaqal_v1_3.dump_insts",
                                                        args + 1, nargs - 1, kwnames);
            bool pfloat = true;
            if (_pfloat)
                pfloat = py::arg_cast<py::bool_,true>(_pfloat, "print_float").as_bool();
            return print_inst(std::span(b.data(), b.size()), pfloat);
        },"",METH_STATIC>,
        py::meth_fastkw<"extract_pulses",[] (auto, PyObject *const *args, Py_ssize_t nargs,
                                             py::tuple kwnames) {
            py::check_num_arg("Jaqal_v1_3.extract_pulses", nargs, 1, 2);
            auto b = py::arg_cast<py::bytes>(args[0], "b");
            auto [_single] =
                py::parse_pos_or_kw_args<"single_action">("Jaqal_v1_3.dump_insts",
                                                          args + 1, nargs - 1, kwnames);
            bool single = true;
            if (_single)
                single = py::arg_cast<py::bool_,true>(_single, "single_action").as_bool();
            PulseSequencer sequencer(single);
            Executor::execute(sequencer, std::span(b.data(), b.size()));
            return py::new_nlist(sequencer.pulses.size(), [&] (int i) {
                return alloc<PyInst>(sequencer.pulses[i]);
            });
        },"",METH_STATIC>>),
};

} // (anonymous)

} // Jaqal_v1_3

PyTypeObject &JaqalInst_v1_Type = Jaqal_v1::PyInst::Type;
PyTypeObject &Jaqal_v1_Type = Jaqal_v1::PyJaqal::Type;
PyTypeObject &JaqalChannelGen_v1_Type = Jaqal_v1::PyChannelGen::Type;
PyTypeObject &JaqalInst_v1_3_Type = Jaqal_v1_3::PyInst::Type;
PyTypeObject &Jaqal_v1_3_Type = Jaqal_v1_3::PyJaqal::Type;

__attribute__((visibility("hidden")))
void init()
{
    throw_if(PyType_Ready(&PyJaqalInstBase::Type) < 0);
    throw_if(PyType_Ready(&Jaqal_v1::PyInst::Type) < 0);
    throw_if(PyType_Ready(&Jaqal_v1::PyJaqal::Type) < 0);
    throw_if(PyType_Ready(&Jaqal_v1::PyChannelGen::Type) < 0);
    throw_if(PyType_Ready(&Jaqal_v1_3::PyInst::Type) < 0);
    throw_if(PyType_Ready(&Jaqal_v1_3::PyJaqal::Type) < 0);
}

}
