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

#include <bitset>
#include <cctype>

namespace brassboard_seq::rfsoc_backend {

__attribute__((visibility("protected")))
void ChannelInfo::ensure_unused_tones(bool all)
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

static int parse_pos_int(const std::string_view &s, py::tuple path, int max)
{
    if (!std::ranges::all_of(s, [] (auto c) { return std::isdigit(c); }))
        config::raise_invalid_channel(path);
    int n;
    if (std::from_chars(s.data(), s.data() + s.size(), n).ec != std::errc{} || n > max)
        config::raise_invalid_channel(path);
    return n;
}

__attribute__((visibility("protected")))
void ChannelInfo::collect_channel(py::ptr<seq::Seq> seq, py::str prefix)
{
    // Channel name format: <prefix>/dds<chn>/<tone>/<param>
    std::map<int,int> chn_idx_map;
    for (auto [idx, path]: py::list_iter<py::tuple>(seq->seqinfo->channel_paths)) {
        if (path.get<py::str>(0).compare(prefix) != 0)
            continue;
        if (path.size() != 4)
            config::raise_invalid_channel(path);
        auto ddspath = path.get<py::str>(1).utf8_view();
        if (!ddspath.starts_with("dds"))
            config::raise_invalid_channel(path);
        ddspath.remove_prefix(3);
        auto chn = ((parse_pos_int(ddspath, path, 31) << 1) |
                    parse_pos_int(path.get<py::str>(2).utf8_view(), path, 1));
        int chn_idx;
        if (auto it = chn_idx_map.find(chn); it == chn_idx_map.end()) {
            chn_idx = (int)channels.size();
            channels.push_back({ chn });
            chn_idx_map[chn] = chn_idx;
        }
        else {
            chn_idx = it->second;
        }
        auto param = path.get<py::str>(3);
        if (param.compare_ascii("freq") == 0) {
            chn_map.insert({(int)idx, {chn_idx, ToneFreq}});
        }
        else if (param.compare_ascii("phase") == 0) {
            chn_map.insert({(int)idx, {chn_idx, TonePhase}});
        }
        else if (param.compare_ascii("amp") == 0) {
            chn_map.insert({(int)idx, {chn_idx, ToneAmp}});
        }
        else if (param.compare_ascii("ff") == 0) {
            chn_map.insert({(int)idx, {chn_idx, ToneFF}});
        }
        else {
            config::raise_invalid_channel(path);
        }
    }
}

}
