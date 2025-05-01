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

#include "action.h"
#include "config.h"
#include "event_time.h"
#include "rfsoc.h"
#include "rtprop.h"
#include "rtval.h"
#include "scan.h"
#include "seq.h"
#include "yaml.h"

#include <mutex>

namespace brassboard_seq {

static std::once_flag init_flag;

void init()
{
    std::call_once(init_flag, [] {
        rtval::init();
        action::init();
        config::init();
        event_time::init();
        rfsoc::init();
        rtprop::init();
        scan::init();
        seq::init();
        yaml::init();
        return 0;
    });
}

}
