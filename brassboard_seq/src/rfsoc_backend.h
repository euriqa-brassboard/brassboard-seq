//

#ifndef BRASSBOARD_SEQ_SRC_RFSOC_BACKEND_H
#define BRASSBOARD_SEQ_SRC_RFSOC_BACKEND_H

namespace rfsoc_backend {

struct cubic_spline_t {
    double order0;
    double order1;
    double order2;
    double order3;
    bool operator==(const cubic_spline_t &other) const
    {
        return (order0 == other.order0) && (order1 == other.order1) &&
            (order2 == other.order2) && (order3 == other.order3);
    }
};

struct output_flags_t {
    bool wait_trigger;
    bool sync;
    bool feedback_enable;
};

}

#endif
