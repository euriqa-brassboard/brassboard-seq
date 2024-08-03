//

#ifndef BRASSBOARD_SEQ_SRC_EVENT_TIME_H
#define BRASSBOARD_SEQ_SRC_EVENT_TIME_H

namespace brassboard_seq::event_time {

static constexpr long long time_scale = 1000000000000ll;

struct EventTimeData {
    int id;
    bool floating: 1;
    int chain_id: 31;
    bool has_static: 1;
    uint64_t c_offset: 63;

    inline long long _get_static() const
    {
        return (long long)c_offset;
    }
    inline long long get_static() const
    {
        if (has_static)
            return (long long)c_offset;
        return -1;
    }

    inline void set_static(long long value)
    {
        has_static = true;
        c_offset = (uint64_t)value;
    }
};

}

#endif
