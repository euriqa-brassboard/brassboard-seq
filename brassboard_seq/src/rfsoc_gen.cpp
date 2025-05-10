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

static inline cubic_spline approximate_spline(double v[5])
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
    return cubic_spline::from_values(v0, v1, v2, v3);
}

__attribute__((visibility("internal")))
inline void Jaqalv1Gen::process_channel(ToneBuffer &tone_buffer, int chn,
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
                return action.spline.resample(t1, t2);
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
                    freqs[cycle] = freq_action.eval(cur_cycle + cycle - freq_cycle);
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
                    phases[cycle] = phase_action.eval(cur_cycle + cycle - phase_cycle);
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
                    amps[cycle] = amp_action.eval(cur_cycle + cycle - amp_cycle);
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

__attribute__((visibility("internal")))
inline void Jaqalv1Gen::BoardGen::clear()
{
    for (auto &channel: channels) {
        channel.clear();
    }
}

__attribute__((visibility("internal")))
inline void Jaqalv1Gen::start()
{
    for (auto &board: boards) {
        board.clear();
    }
}

__attribute__((visibility("internal")))
inline void Jaqalv1Gen::end()
{
#pragma omp parallel
#pragma omp single
#pragma omp taskloop
    for (auto &board: boards) {
        board.end();
    }
}

__attribute__((visibility("protected")))
py::ref<> Jaqalv1Gen::get_prefix(int n) const
{
    if (n < 0 || n >= 4)
        throw std::out_of_range("Board index should be in [0, 3]");
    return boards[n].get_prefix();
}

__attribute__((visibility("protected")))
py::ref<> Jaqalv1Gen::get_sequence(int n) const
{
    if (n < 0 || n >= 4)
        throw std::out_of_range("Board index should be in [0, 3]");
    return boards[n].get_sequence();
}

__attribute__((flatten))
static inline void chn_add_tone_data(auto &channel_gen, int channel, int tone,
                                     int64_t duration_cycles,
                                     cubic_spline freq, cubic_spline amp,
                                     cubic_spline phase, output_flags_t flags,
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
inline void Jaqalv1Gen::add_tone_data(int chn, int64_t duration_cycles,
                                      cubic_spline freq, cubic_spline amp,
                                      cubic_spline phase,
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
            return spline.resample_cycle(0, duration_cycles, tstart, tend);
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
inline py::ref<> Jaqalv1Gen::BoardGen::get_prefix() const
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
inline py::ref<> Jaqalv1Gen::BoardGen::get_sequence() const
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
inline void Jaqalv1Gen::BoardGen::end()
{
#pragma omp taskloop
    for (auto &channel_gen: channels) {
        channel_gen.end();
    }
}

inline __attribute__((always_inline))
void Jaqalv1_3Gen::process_freq(std::span<DDSParamAction> freq_actions,
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
                return action.spline.resample(t1, t2);
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
                    freqs[cycle] = freq_action.eval(cur_cycle + cycle - freq_cycle);
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

template<auto pulsef>
inline __attribute__((always_inline))
void Jaqalv1_3Gen::process_param(std::span<DDSParamAction> actions, ChnInfo chn,
                                 int64_t total_cycle, Jaqal_v1_3::ModType modtype)
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
                return action.spline.resample(t1, t2);
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

            // Now we don't have enough time to do a tone data
            // based on the segmentation given to us. We'll manually iterate over
            // the next 4 cycles and compute a 4 cycle tone data that approximate
            // the action we need the closest.

            double params[5];
            while (true) {
                auto min_cycle = (int)std::max(action_cycle - cur_cycle, int64_t(0));
                auto max_cycle = (int)std::min(action_end_cycle - cur_cycle, int64_t(4));
                for (int cycle = min_cycle; cycle <= max_cycle; cycle++)
                    params[cycle] = action.eval(cur_cycle + cycle - action_cycle);
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
void Jaqalv1_3Gen::process_frame(ChnInfo chn, int64_t total_cycle,
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
inline void Jaqalv1_3Gen::process_channel(ToneBuffer &tone_buffer, int chn,
                                          int64_t total_cycle)
{
    bb_debug("Start outputting jaqal v1.3 insts for channel %d\n", chn);
    assert(!tone_buffer.params[0].empty());
    assert(!tone_buffer.params[1].empty());
    assert(!tone_buffer.params[2].empty());

    ChnInfo chninfo{chn};
    process_freq(tone_buffer.params[(int)ToneFreq], tone_buffer.ff,
                 chninfo, total_cycle);
    process_param<Jaqal_v1_3::amp_pulse>(
        tone_buffer.params[(int)ToneAmp], chninfo, total_cycle,
        chninfo.tone == 0 ? Jaqal_v1_3::AMPMOD0 : Jaqal_v1_3::AMPMOD1);
    process_param<Jaqal_v1_3::phase_pulse>(
        tone_buffer.params[(int)TonePhase], chninfo, total_cycle,
        chninfo.tone == 0 ? Jaqal_v1_3::PHSMOD0 : Jaqal_v1_3::PHSMOD1);
    process_frame(chninfo, total_cycle,
                  chninfo.tone == 0 ? Jaqal_v1_3::FRMROT0 : Jaqal_v1_3::FRMROT1);
}

__attribute__((visibility("protected")))
py::ref<> Jaqalv1_3Gen::get_prefix(int n) const
{
    if (n < 0 || n >= 4)
        throw std::out_of_range("Board index should be in [0, 3]");
    return py::new_bytes();
}

__attribute__((visibility("protected")))
py::ref<> Jaqalv1_3Gen::get_sequence(int n) const
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
    return res;
}

__attribute__((visibility("internal")))
inline void Jaqalv1_3Gen::add_inst(const JaqalInst &inst, int board, int board_chn,
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
inline void Jaqalv1_3Gen::start()
{
    for (auto &insts: board_insts) {
        insts.clear();
    }
}

__attribute__((visibility("internal")))
inline void Jaqalv1_3Gen::end()
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

__attribute__((visibility("protected"))) void
SyncTimeMgr::add_action(std::vector<DDSParamAction> &actions, int64_t start_cycle,
                        int64_t end_cycle, cubic_spline sp,
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
            sync_freq = sp.eval(t);
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
                sp.resample_cycle(start_cycle, end_cycle, start_cycle, sync_cycle) });
    } else if (sync_freq != sp.order0) {
        // We have a sync at frequency action boundary.
        // This is the only case we may need to sync at a different frequency
        // compared to the frequency of the output immediately follows this.
        bb_debug("  0-length sync @%" PRId64 "\n", start_cycle);
        actions.push_back({ 0, true, cubic_spline::from_static(sync_freq) });
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
                    sp.resample_cycle(start_cycle, end_cycle, sync_cycle, end_cycle) });
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
                sp.resample_cycle(start_cycle, end_cycle, prev_cycle, sync_cycle) });
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
            sync_freq = sp.eval(t);
        }
        bb_debug("  updated sync frequency: %f @%" PRId64 ", sync_freq_match_tid: %d\n",
                 sync_freq, sync_freq_seq_time, sync_freq_match_tid);
    }
}

__attribute__((visibility("protected")))
PyTypeObject RFSOCGenerator::Type = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "brassboard_seq.rfsoc_backend.RFSOCGenerator",
    .tp_basicsize = sizeof(RFSOCGenerator),
    .tp_dealloc = py::tp_cxx_dealloc<false,RFSOCGenerator>,
    .tp_flags = Py_TPFLAGS_DEFAULT,
};

namespace {

struct Jaqalv1Generator : RFSOCGenerator {
    static PyTypeObject Type;
};

PyTypeObject Jaqalv1Generator::Type = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "brassboard_seq.rfsoc_backend.Jaqalv1Generator",
    .tp_basicsize = sizeof(Jaqalv1Generator),
    .tp_dealloc = py::tp_cxx_dealloc<false,Jaqalv1Generator>,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_methods = (
        py::meth_table<
        py::meth_o<"get_prefix",[] (py::ptr<Jaqalv1Generator> self, py::ptr<> _n) {
            auto n = py::arg_cast<py::int_>(_n, "n").as_int();
            return ((Jaqalv1Gen*)self->gen.get())->get_prefix(n);
        }>,
        py::meth_o<"get_sequence",[] (py::ptr<Jaqalv1Generator> self, py::ptr<> _n) {
            auto n = py::arg_cast<py::int_>(_n, "n").as_int();
            return ((Jaqalv1Gen*)self->gen.get())->get_sequence(n);
        }>>),
    .tp_base = &RFSOCGenerator::Type,
    .tp_vectorcall = py::vectorfunc<[] (auto, PyObject *const *args,
                                        ssize_t nargs, py::tuple kwnames) {
        py::check_num_arg("Jaqalv1Generator.__init__", nargs, 0, 0);
        py::check_no_kwnames("Jaqalv1Generator.__init__", kwnames);
        auto self = py::generic_alloc<Jaqalv1Generator>();
        self->gen.reset(new Jaqalv1Gen);
        return self;
    }>
};

struct Jaqalv1_3Generator : RFSOCGenerator {
    static PyTypeObject Type;
};

PyTypeObject Jaqalv1_3Generator::Type = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "brassboard_seq.rfsoc_backend.Jaqalv1_3Generator",
    .tp_basicsize = sizeof(Jaqalv1_3Generator),
    .tp_dealloc = py::tp_cxx_dealloc<false,Jaqalv1_3Generator>,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_methods = (
        py::meth_table<
        py::meth_o<"get_prefix",[] (py::ptr<Jaqalv1_3Generator> self, py::ptr<> _n) {
            auto n = py::arg_cast<py::int_>(_n, "n").as_int();
            return ((Jaqalv1_3Gen*)self->gen.get())->get_prefix(n);
        }>,
        py::meth_o<"get_sequence",[] (py::ptr<Jaqalv1_3Generator> self, py::ptr<> _n) {
            auto n = py::arg_cast<py::int_>(_n, "n").as_int();
            return ((Jaqalv1_3Gen*)self->gen.get())->get_sequence(n);
        }>>),
    .tp_base = &RFSOCGenerator::Type,
    .tp_vectorcall = py::vectorfunc<[] (auto, PyObject *const *args,
                                        ssize_t nargs, py::tuple kwnames) {
        py::check_num_arg("Jaqalv1_3Generator.__init__", nargs, 0, 0);
        py::check_no_kwnames("Jaqalv1_3Generator.__init__", kwnames);
        auto self = py::generic_alloc<Jaqalv1_3Generator>();
        self->gen.reset(new Jaqalv1_3Gen);
        return self;
    }>
};

} // (anonymous)

PyTypeObject &Jaqalv1Generator_Type = Jaqalv1Generator::Type;
PyTypeObject &Jaqalv1_3Generator_Type = Jaqalv1_3Generator::Type;

__attribute__((visibility("hidden")))
void init()
{
    throw_if(PyType_Ready(&RFSOCGenerator::Type) < 0);
    throw_if(PyType_Ready(&Jaqalv1Generator::Type) < 0);
    throw_if(PyType_Ready(&Jaqalv1_3Generator::Type) < 0);
}

}
