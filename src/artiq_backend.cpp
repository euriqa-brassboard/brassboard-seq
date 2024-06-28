//

#include "artiq_backend.h"

#include <algorithm>

namespace artiq_backend {
static ArtiqConsts artiq_consts;

inline void UrukulBus::config_and_write(std::vector<RTIOAction> &actions,
                                        uint32_t flags, uint32_t length,
                                        uint32_t cs, uint32_t data)
{
    auto div = artiq_consts.URUKUL_SPIT_DDS_WR;
    auto addr = flags | ((length - 1) << 8) | ((div - 2) << 16) | (cs << 24);
    uint16_t data_time_mu = uint16_t(((length + 1) * div + 1) * ref_period_mu);
    int idx0 = (int)actions.size();
    actions.push_back(RTIOAction{addr_target, addr, last_idx, last_gap_mu,
            false, true, -1, -1});
    actions.push_back(RTIOAction{data_target, data, idx0, ref_period_mu,
            false, true, -1, -1});
    last_idx = idx0 + 1;
    last_gap_mu = data_time_mu;
}

void UrukulBus::write_set(std::vector<RTIOAction> &actions,
                          uint32_t cs, uint32_t data1, uint32_t data2)
{
    config_and_write(actions, artiq_consts.URUKUL_CONFIG, 8, cs,
                     artiq_consts._AD9910_REG_PROFILE0 << 24);
    config_and_write(actions, artiq_consts.URUKUL_CONFIG, 32, cs, data1);
    config_and_write(actions, artiq_consts.URUKUL_CONFIG_END, 32, cs, data2);
}

inline void TimeChecker::clear()
{
    counts.clear();
    max_key = 0;
}

inline bool TimeChecker::check_and_add_time(long long t_mu)
{
    auto t_course = t_mu >> 3; // hard code
    if (t_course > max_key) {
        max_key = t_course;
        counts[t_course] = 1;
        return true;
    }
    auto &cnt = counts[t_course];
    if (cnt >= 7)
        return false;
    cnt += 1;
    return true;
}

inline long long TimeChecker::find_time(long long lb_mu, long long t_mu,
                                        long long ub_mu)
{
    if (t_mu > 0 && t_mu > lb_mu) {
        // There is a desired time, try to honor it as much as possible
        auto orig_t_mu = t_mu;
        while (t_mu >= lb_mu) {
            if (check_and_add_time(t_mu))
                return t_mu;
            t_mu -= 8;
        }
        t_mu = orig_t_mu + 8;
    }
    else {
        t_mu = lb_mu;
    }
    while (t_mu <= ub_mu) {
        if (check_and_add_time(t_mu))
            return t_mu;
        t_mu += 8;
    }
    return -1;
}

std::tuple<ScheduleErrorCode,int> RTIOScheduler::schedule(uint64_t total_time_mu)
{
    auto nactions = (int)actions.size();
    checker.clear();
    for (auto &action: actions) {
        // TODO: remove?
        if (action.exact_time) {
            action.time_ub_mu = action.request_mu;
        }
        else {
            action.time_ub_mu = total_time_mu + 1000; // 1 us buffer
        }
    }

    // Setting the upper bound
    for (int i = nactions - 1; i >= 0; i--) {
        auto &action = actions[i];
        auto lb_action_idx = action.lb_action_idx;
        if (lb_action_idx != -1) {
            auto time_ub_mu = action.time_ub_mu;
            auto lb_action = actions[lb_action_idx];
            time_ub_mu -= action.lb_offset_mu;
            lb_action.time_ub_mu = std::min(lb_action.time_ub_mu, time_ub_mu);
        }
    }

    for (int i = 0; i < nactions; i++) {
        auto &action = actions[i];
        int64_t lb_mu = 0;
        if (action.lb_action_idx != -1)
            lb_mu = actions[action.lb_action_idx].request_mu + action.lb_offset_mu;
        auto ub_mu = action.time_ub_mu;
        if (action.exact_time) {
            if (action.request_mu < lb_mu)
                return std::make_tuple(ExactLB, i);
            if (action.request_mu > ub_mu)
                return std::make_tuple(ExactUB, i);
            if (!checker.check_and_add_time(action.request_mu))
                return std::make_tuple(TooManyPulses, i);
            continue;
        }
        if (action.coarse_align) {
            lb_mu = ((lb_mu - 1) & ~int64_t(7)) + 8;
            ub_mu = ub_mu & ~int64_t(7);
        }
        if (lb_mu > ub_mu)
            return std::make_tuple(LBUB, i);

        if (action.request_mu != -1) {
            lb_mu = std::max(lb_mu, action.request_mu - 3500); // 3.5 us
            ub_mu = std::min(ub_mu, action.request_mu + 3500); // 3.5 us
        }

        if (lb_mu > ub_mu)
            return std::make_tuple(FarFromRequest, i);

        auto actual_mu = checker.find_time(lb_mu, action.request_mu, ub_mu);
        if (actual_mu == -1)
            return std::make_tuple(TooManyPulses, i);
        action.request_mu = actual_mu;
    }
    return std::make_tuple(NoError, 0);
}

}
