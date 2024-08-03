//

#ifndef BRASSBOARD_SEQ_SRC_ACTION_H
#define BRASSBOARD_SEQ_SRC_ACTION_H

namespace brassboard_seq::action {

struct ActionData {
    bool is_pulse;
    bool exact_time;
    bool cond_val;
};

}

#endif
