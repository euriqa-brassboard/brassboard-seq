//

#ifndef BRASSBOARD_SEQ_SRC_ARTIQ_BACKEND_H
#define BRASSBOARD_SEQ_SRC_ARTIQ_BACKEND_H

#include <tuple>
#include <unordered_map>
#include <vector>

#include <stdint.h>

namespace artiq_backend {

struct ArtiqConsts {
    int COUNTER_ENABLE;
    int COUNTER_DISABLE;
    int _AD9910_REG_PROFILE0;
    int URUKUL_CONFIG;
    int URUKUL_CONFIG_END;
    int URUKUL_SPIT_DDS_WR;
    int SPI_CONFIG_ADDR;
    int SPI_DATA_ADDR;
};

struct RTIOAction {
    uint32_t target;
    uint32_t value;
    int32_t lb_action_idx;
    uint16_t lb_offset_mu;
    bool exact_time;
    bool coarse_align;
    int64_t request_mu;
    int64_t time_ub_mu;
};

struct UrukulBus {
    uint32_t channel;
    uint32_t addr_target;
    uint32_t data_target;
    uint32_t io_update_target;
    uint8_t ref_period_mu;

    uint16_t last_gap_mu = 0;
    int last_idx = -1;

    void config_and_write(std::vector<RTIOAction> &actions, uint32_t flags,
                          uint32_t length, uint32_t cs, uint32_t data);
    void write_set(std::vector<RTIOAction> &actions,
                   uint32_t cs, uint32_t data1, uint32_t data2);
};

enum ChannelType : uint8_t {
    DDSFreq,
    DDSAmp,
    DDSPhase,
    TTLOut,
    CounterEnable,
};

struct DDSChannel {
    double ftw_per_hz;
    uint32_t bus_id;
    uint8_t chip_select;
};

struct TTLChannel {
    uint32_t target;
    bool iscounter;
};

struct SeqAction {
    ChannelType param;
    bool exact_time;
    uint16_t artiq_chn;
    uint32_t val;
    uint64_t time_mu;
};

enum ScheduleErrorCode {
    NoError = 0,
    ExactLB = 1,
    ExactUB = 2,
    LBUB = 3,
    FarFromRequest = 4,
    TooManyPulses = 5,
};

struct TimeChecker {
    TimeChecker()
        : max_key(0)
    {}
    void clear();
    bool check_and_add_time(long long t_mu);
    long long find_time(long long lb_mu, long long t_mu, long long ub_mu);

    std::unordered_map<long long,int> counts;
    long long max_key;
};

struct RTIOScheduler {
    std::tuple<ScheduleErrorCode,int> schedule(uint64_t total_time_mu);

    std::vector<RTIOAction> actions;
    TimeChecker checker;
};

}

#endif
