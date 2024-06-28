//

#ifndef BRASSBOARD_SEQ_SRC_ARTIQ_BACKEND_H
#define BRASSBOARD_SEQ_SRC_ARTIQ_BACKEND_H

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

struct UrukulBus {
    uint32_t channel;
    uint32_t addr_target;
    uint32_t data_target;
    uint32_t io_update_target;
    uint8_t ref_period_mu;
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

}

#endif
