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

#include "rfsoc_gen.h"

namespace brassboard_seq::rfsoc_gen {

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

__attribute__((visibility("internal")))
inline void SyncChannelGen::process_channel(ToneBuffer &tone_buffer, int chn,
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

struct PulseCompilerGen::Info {
    PyObject *channel_list[64];
    py::ptr<> CubicSpline;
    py::ptr<> ToneData;
    py::ptr<> cubic_0;
    std::vector<std::pair<PyObject*,PyObject*>> tonedata_fields;

    py::ref<> _new_cubic_spline(cubic_spline_t sp)
    {
        auto newobj = py::generic_alloc<py::tuple>(CubicSpline, 4);
        newobj.SET(0, py::new_float(sp.order0));
        newobj.SET(1, py::new_float(sp.order1));
        newobj.SET(2, py::new_float(sp.order2));
        newobj.SET(3, py::new_float(sp.order3));
        return newobj;
    }

    py::ref<> new_cubic_spline(cubic_spline_t sp)
    {
        if (sp == cubic_spline_t{0, 0, 0, 0})
            return cubic_0.ref();
        return _new_cubic_spline(sp);
    }

    py::ref<> new_tone_data(int channel, int tone, int64_t duration_cycles,
                            cubic_spline_t freq, cubic_spline_t amp,
                            cubic_spline_t phase, output_flags_t flags)
    {
        auto td = py::generic_alloc(ToneData);
        auto td_dict = py::dict_ref(throw_if_not(PyObject_GenericGetDict(td.get(), nullptr)));
        for (auto [key, value]: tonedata_fields)
            td_dict.set(key, value);
        py::assert_int_cache<32>();
        td_dict.set("channel"_py, py::int_cached(channel));
        td_dict.set("tone"_py, py::int_cached(tone));
        td_dict.set("duration_cycles"_py, py::new_int(duration_cycles));
        td_dict.set("frequency_hz"_py, new_cubic_spline(freq));
        td_dict.set("amplitude"_py, new_cubic_spline(amp));
        // tone data wants rad as phase unit.
        td_dict.set("phase_rad"_py, new_cubic_spline({
                    phase.order0 * (2 * M_PI), phase.order1 * (2 * M_PI),
                    phase.order2 * (2 * M_PI), phase.order3 * (2 * M_PI) }));
        td_dict.set("frame_rotation_rad"_py, cubic_0);
        td_dict.set("wait_trigger"_py, flags.wait_trigger ? Py_True : Py_False);
        td_dict.set("sync"_py, flags.sync ? Py_True : Py_False);
        td_dict.set("output_enable"_py, Py_False);
        td_dict.set("feedback_enable"_py, flags.feedback_enable ? Py_True : Py_False);
        td_dict.set("bypass_lookup_tables"_py, Py_False);
        return td;
    }

    Info();
};

__attribute__((visibility("internal")))
inline auto PulseCompilerGen::get_info() -> Info*
{
    static Info info;
    return &info;
}

__attribute__((visibility("internal")))
inline void PulseCompilerGen::add_tone_data(int chn, int64_t duration_cycles,
                                            cubic_spline_t freq, cubic_spline_t amp,
                                            cubic_spline_t phase,
                                            output_flags_t flags, int64_t)
{
    bb_debug("outputting tone data: chn=%d, cycles=%" PRId64 ", sync=%d, ff=%d\n",
             chn, duration_cycles, flags.sync, flags.feedback_enable);
    auto info = get_info();
    auto tonedata = info->new_tone_data(chn >> 1, chn & 1, duration_cycles, freq,
                                        amp, phase, flags);
    auto key = info->channel_list[chn];
    py::list tonedatas;
    if (last_chn == chn) [[likely]] {
        tonedatas = assume(last_tonedatas);
    }
    else {
        tonedatas = output.try_get(key);
    }
    if (!tonedatas) {
        auto tonedatas = py::new_list(std::move(tonedata));
        output.set(key, tonedatas);
        last_tonedatas = tonedatas;
    }
    else {
        py::list(tonedatas).append(std::move(tonedata));
        last_tonedatas = tonedatas;
    }
    last_chn = chn;
}

__attribute__((visibility("internal")))
inline void PulseCompilerGen::start()
{
    output.clear();
    last_chn = -1;
}

__attribute__((visibility("internal")))
inline void PulseCompilerGen::end()
{
}

__attribute__((visibility("protected")))
Generator *new_pulse_compiler_generator()
{
    return new PulseCompilerGen;
}

__attribute__((visibility("internal")))
inline PulseCompilerGen::Info::Info()
{
    auto tonedata_mod = py::import_module("pulsecompiler.rfsoc.tones.tonedata");
    ToneData = tonedata_mod.attr("ToneData").rel();
    auto splines_mod = py::import_module("pulsecompiler.rfsoc.structures.splines");
    CubicSpline = splines_mod.attr("CubicSpline").rel();
    cubic_0 = _new_cubic_spline({0, 0, 0, 0}).rel();
    auto pulse_mod = py::import_module("qiskit.pulse");
    auto ControlChannel = pulse_mod.attr("ControlChannel");
    auto DriveChannel = pulse_mod.attr("DriveChannel");

    py::assert_int_cache<64>();
    PyObject *py_nums[64];
    for (int i = 0; i < 64; i++)
        py_nums[i] = py::int_cached(i);

    channel_list[0] = ControlChannel(py_nums[0]).rel();
    channel_list[1] = ControlChannel(py_nums[1]).rel();
    for (int i = 0; i < 62; i++)
        channel_list[i + 2] = DriveChannel(py_nums[i]).rel();

    auto orig_post_init = ToneData.attr("__post_init__");
    static PyMethodDef dummy_post_init_method = py::meth_fast<"__post_init__",[] (auto...) {}>;
    ToneData.set_attr("__post_init__", py::new_cfunc(&dummy_post_init_method));
    auto dummy_tonedata = ToneData(py_nums[0], py_nums[0], py_nums[0],
                                   py_nums[0], py_nums[0], py_nums[0]);
    ToneData.set_attr("__post_init__", orig_post_init);
    auto td_dict = py::dict_ref::checked(PyObject_GenericGetDict(dummy_tonedata.get(),
                                                                 nullptr));
    for (auto [key, value]: py::dict_iter<PyObject,py::str>(td_dict)) {
        for (auto name: {"channel", "tone", "duration_cycles", "frequency_hz",
                "amplitude", "phase_rad", "frame_rotation_rad", "wait_trigger",
                "sync", "output_enable", "feedback_enable",
                "bypass_lookup_tables"}) {
            if (key.compare_ascii(name) == 0) {
                goto skip_key;
            }
        }
        tonedata_fields.push_back({ py::newref(key), py::newref(value) });
    skip_key:
        ;
    }
}

__attribute__((visibility("internal")))
inline void JaqalPulseCompilerGen::BoardGen::clear()
{
    for (auto &channel: channels) {
        channel.clear();
    }
}

__attribute__((visibility("internal")))
inline void JaqalPulseCompilerGen::start()
{
    for (auto &board: boards) {
        board.clear();
    }
}

__attribute__((visibility("internal")))
inline void JaqalPulseCompilerGen::end()
{
#pragma omp parallel
#pragma omp single
#pragma omp taskloop
    for (auto &board: boards) {
        board.end();
    }
}

__attribute__((visibility("protected")))
PyObject *JaqalPulseCompilerGen::get_prefix(int n) const
{
    if (n < 0 || n >= 4)
        throw std::out_of_range("Board index should be in [0, 3]");
    return boards[n].get_prefix();
}

__attribute__((visibility("protected")))
PyObject *JaqalPulseCompilerGen::get_sequence(int n) const
{
    if (n < 0 || n >= 4)
        throw std::out_of_range("Board index should be in [0, 3]");
    return boards[n].get_sequence();
}

__attribute__((visibility("protected")))
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

__attribute__((visibility("internal")))
inline void JaqalPulseCompilerGen::add_tone_data(int chn, int64_t duration_cycles,
                                                 cubic_spline_t freq, cubic_spline_t amp,
                                                 cubic_spline_t phase,
                                                 output_flags_t flags, int64_t cur_cycle)
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

__attribute__((visibility("internal")))
inline PyObject *JaqalPulseCompilerGen::BoardGen::get_prefix() const
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

__attribute__((visibility("internal")))
inline PyObject *JaqalPulseCompilerGen::BoardGen::get_sequence() const
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

__attribute__((visibility("internal")))
inline void JaqalPulseCompilerGen::BoardGen::end()
{
#pragma omp taskloop
    for (auto &channel_gen: channels) {
        channel_gen.end();
    }
}

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

__attribute__((visibility("internal")))
inline void Jaqalv1_3Generator::process_channel(ToneBuffer &tone_buffer, int chn,
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

__attribute__((visibility("protected")))
PyObject *Jaqalv1_3StreamGen::get_prefix(int n) const
{
    if (n < 0 || n >= 4)
        throw std::out_of_range("Board index should be in [0, 3]");
    return py::empty_bytes.immref().rel();
}

__attribute__((visibility("protected")))
PyObject *Jaqalv1_3StreamGen::get_sequence(int n) const
{
    if (n < 0 || n >= 4)
        throw std::out_of_range("Board index should be in [0, 3]");
    auto &insts = board_insts[n];
    auto ninsts = insts.size();
    static constexpr auto instsz = sizeof(JaqalInst);
    auto res = py::new_bytes(nullptr, ninsts * instsz);
    auto ptr = res.data();
    for (size_t i = 0; i < ninsts; i++)
        memcpy(&ptr[i * instsz], &insts[i].inst, instsz);
    return res.rel();
}

__attribute__((visibility("internal")))
inline void Jaqalv1_3StreamGen::add_inst(const JaqalInst &inst, int board, int board_chn,
                                         Jaqal_v1_3::ModType mod, int64_t cycle)
{
    assert(board >= 0 && board < 4);
    auto &insts = board_insts[board];
    auto real_inst = Jaqal_v1_3::apply_channel_mask(inst, 1 << board_chn);
    real_inst = Jaqal_v1_3::apply_modtype_mask(real_inst,
                                               Jaqal_v1_3::ModTypeMask(1 << mod));
    insts.push_back({ cycle, Jaqal_v1_3::stream(real_inst) });
}

__attribute__((visibility("internal")))
inline void Jaqalv1_3StreamGen::start()
{
    for (auto &insts: board_insts) {
        insts.clear();
    }
}

__attribute__((visibility("internal")))
inline void Jaqalv1_3StreamGen::end()
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

__attribute__((visibility("protected")))
Generator *new_jaqalv1_3_stream_generator()
{
    return new Jaqalv1_3StreamGen;
}

__attribute__((visibility("protected"))) void
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

}
