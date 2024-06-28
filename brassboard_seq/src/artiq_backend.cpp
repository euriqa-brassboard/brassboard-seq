//

#include "artiq_backend.h"

#include <assert.h>

namespace artiq_backend {
static ArtiqConsts artiq_consts;

inline int ChannelsInfo::add_bus_channel(int bus_channel, uint32_t io_update_target,
                                         uint8_t ref_period_mu)
{
    auto bus_id = (int)urukul_busses.size();
    urukul_busses.push_back({
            uint32_t(bus_channel),
            uint32_t((bus_channel << 8) | artiq_consts.SPI_CONFIG_ADDR),
            uint32_t((bus_channel << 8) | artiq_consts.SPI_DATA_ADDR),
            // Here we assume that the CPLD (and it's io_update channel)
            // and the SPI bus has a one-to-one mapping.
            // This means that each DDS with the same bus shares
            // the same io_update channel and can only be programmed one at a time.
            io_update_target,
            ref_period_mu,
        });
    bus_chn_map[bus_channel] = bus_id;
    return bus_id;
}

inline void ChannelsInfo::add_ttl_channel(int seqchn, uint32_t target, bool iscounter)
{
    assert(ttl_chn_map.count(seqchn) == 0);
    auto ttl_id = (int)ttlchns.size();
    ttlchns.push_back({target, iscounter});
    ttl_chn_map[seqchn] = ttl_id;
}

inline int ChannelsInfo::get_dds_channel_id(uint32_t bus_id, double ftw_per_hz,
                                            uint8_t chip_select)
{
    std::pair<int,int> key{bus_id, chip_select};
    auto it = dds_chn_map.find(key);
    if (it != dds_chn_map.end())
        return it->second;
    auto dds_id = (int)ddschns.size();
    ddschns.push_back({ftw_per_hz, bus_id, chip_select});
    dds_chn_map[key] = dds_id;
    return dds_id;
}

inline void ChannelsInfo::add_dds_param_channel(int seqchn, uint32_t bus_id,
                                                double ftw_per_hz, uint8_t chip_select,
                                                ChannelType param)
{
    assert(dds_param_chn_map.count(seqchn) == 0);
    dds_param_chn_map[seqchn] = {get_dds_channel_id(bus_id, ftw_per_hz,
                                                    chip_select), param};
}

}
