//

#ifndef BRASSBOARD_SEQ_SRC_ARTIQ_BACKEND_H
#define BRASSBOARD_SEQ_SRC_ARTIQ_BACKEND_H

#include <stdint.h>

#include <map>
#include <utility>
#include <vector>

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

struct UrukulBus {
    uint32_t channel;
    uint32_t addr_target;
    uint32_t data_target;
    uint32_t io_update_target;
    uint8_t ref_period_mu;
};

struct TTLChannel {
    uint32_t target;
    bool iscounter;
};

struct ChannelsInfo {
    std::vector<UrukulBus> urukul_busses;
    std::vector<TTLChannel> ttlchns;
    std::vector<DDSChannel> ddschns;

    // From bus channel to urukul bus index
    std::map<int,int> bus_chn_map;
    // From sequence channel id to ttl channel index
    std::map<int,int> ttl_chn_map;
    // From (bus_id, chip select) to dds channel index
    std::map<std::pair<int,int>,int> dds_chn_map;
    // From sequence channel id to dds channel index + channel type
    std::map<int,std::pair<int,ChannelType>> dds_param_chn_map;

    ChannelsInfo() = default;
    ChannelsInfo(const ChannelsInfo&) = delete;
    ChannelsInfo(ChannelsInfo&&) = delete;

    inline int find_bus_id(int bus_channel)
    {
        auto it = bus_chn_map.find(bus_channel);
        if (it == bus_chn_map.end())
            return -1;
        return it->second;
    }
    int add_bus_channel(int bus_channel, uint32_t io_update_target,
                        uint8_t ref_period_mu);
    void add_ttl_channel(int seqchn, uint32_t target, bool iscounter);
    int get_dds_channel_id(uint32_t bus_id, double ftw_per_hz, uint8_t chip_select);
    void add_dds_param_channel(int seqchn, uint32_t bus_id, double ftw_per_hz,
                               uint8_t chip_select, ChannelType param);
};

}

#endif
