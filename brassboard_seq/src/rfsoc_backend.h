//

#ifndef BRASSBOARD_SEQ_SRC_RFSOC_BACKEND_H
#define BRASSBOARD_SEQ_SRC_RFSOC_BACKEND_H

#include <map>
#include <vector>
#include <utility>

#include <stdint.h>

#include <Python.h>

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

enum ToneParam {
    ToneFreq,
    TonePhase,
    ToneAmp,
    ToneFF,
    _NumToneParam
};

struct RFSOCAction {
    mutable bool cond;
    mutable bool eval_status;
    bool isramp;
    bool sync;
    int reloc_id;
    int aid;
    int tid: 31;
    bool is_end: 1;
    mutable int64_t seq_time;
    union {
        mutable double float_value;
        mutable bool bool_value;
    };
    PyObject *ramp;
};

struct Relocation {
    // If a particular relocation is not needed for this action,
    // the corresponding idx would be -1
    int cond_idx;
    int time_idx;
    int val_idx;
};

struct ToneChannel {
    ToneChannel(int chn)
        : chn(chn)
    {
    }

    int chn; // ddsnum << 1 | tone
    std::vector<RFSOCAction> actions[4];
};

struct ChannelInfo {
    std::vector<ToneChannel> channels;
    // map from sequence channel to tone channel index
    std::map<int,std::pair<int,ToneParam>> chn_map;

    int add_tone_channel(int chn);
    void add_seq_channel(int seq_chn, int chn_idx, ToneParam param);
};

}

#endif
